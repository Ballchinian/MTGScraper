# Cardalike

A web app that finds Magic: The Gathering cards that do similar things to the card you search for. Not cards that share words, and not "cards that go in the same deck", but cards whose abilities mean the same thing even when the wording is completely different, all by comparing sentence embeddings of every line of rules text ever printed.

## Features

- Card search by name with forgiving matching
- Similarity results ranked by best matching line of rules text
- Every result shows the exact line that matched and a similarity percent
- Load more button that pulls the next 20 results without a page reload
- Scryfall styled interface with card images linking back to Scryfall
- Updates itself! A bot checks Scryfall every day and new cards just appear

## How it works

Everything lives in a Postgres database with the pgvector extension. The embeddings sit in a `vector(384)` column and the database does the nearest neighbor math itself, so the website is a tiny Flask app that runs a few queries per search. The language model only ever runs inside the update pipeline, never on the server.

Three pieces:

| Piece | What it does |
| --- | --- |
| `web/` | The Flask site. Talks to Postgres, needs no torch, no giant files, barely any memory |
| `ingest/` | The update pipeline. Downloads Scryfall's bulk data, embeds whatever is new or changed, writes it to the database |
| `.github/workflows/update.yml` | Runs the pipeline every day on GitHub Actions so the heavy dependencies never go anywhere near the web server |

`common/` holds the bits both sides share: the database schema and the card cleaning helpers.

### The database

Four tables. `cards` has one row per unique card, keyed by Scryfall's `oracle_id` (stable across every printing). `lines` has one row per line of rules text with its embedding. `line_stats` counts how many cards share each exact line, for the ranking weights. `meta` remembers which Scryfall bulk file was processed last.

There is deliberately no vector index: at ~61k rows Postgres scans everything in a few milliseconds and the results are exact, identical to the old in-memory version. If the game ever grows 10x there is a commented out HNSW index in `common/schema.sql` ready to go.

### Daily updates

The pipeline is built so that doing nothing costs nothing:

1. Ask Scryfall's bulk data API for the Oracle Cards file's `updated_at` timestamp (one tiny request).
2. If it matches the one stored in `meta`, stop. Done in two seconds.
3. Otherwise download the bulk file and hash every card's name + rules text. Cards whose hash matches the database get skipped without embedding anything.
4. Only genuinely new or changed cards (usually a handful, or zero) get their lines embedded and written, all inside one transaction.
5. The line counts get rebuilt and the new timestamp saved.

Running the same pipeline against an empty database seeds the whole thing, everything counts as new and ~61k lines get embedded in one big batch. That is the entire initial setup.

Cards basically never get deleted from Scryfall, so the pipeline doesn't bother pruning. Rows for removed cards would just sit there unused.

### Card data

Card data comes from Scryfall's bulk data API, which publishes a daily "Oracle Cards" file with exactly one entry per unique card (Scryfall asks tools to use this instead of scraping pages). Requests send a custom User Agent like their docs require. Cards are filtered before indexing:

| Filter | Why |
| --- | --- |
| Joke sets (funny / memorabilia) | Not real cards |
| Tokens, emblems, art cards, schemes | Not playable cards |
| No rules text | Vanilla creatures and basic lands have nothing to compare |

### Text cleaning

Each line of a card's rules text is treated as one ability, so lines get embedded separately and one matching ability is enough. Before embedding, every line is cleaned: reminder text in parentheses is stripped, and the card's references to its own name are swapped for "this card" so names can't influence matching (including the shortened first name that legendary cards use mid sentence).

### Embeddings

Every cleaned line goes through the `all-MiniLM-L6-v2` sentence transformer, which turns text into a normalized vector of 384 numbers where lines that mean similar things land close together. That is what lets "you may draw a card unless that player pays {4}" match "they may pay {1}. If the player does, they draw a card".

### Ranking

A search grabs the card's own lines from the database, runs a pgvector nearest neighbor query for each one (`<=>` is cosine distance, and the vectors are normalized so `1 - distance` is the real similarity), and keeps each candidate card's best matching pair. Common lines get weighted down so they don't drown out the interesting matches:

| Line | Rough count | Effect on ranking |
| --- | --- | --- |
| "Flying" | Thousands of cards | Heavily downweighted |
| A wordy triggered ability | A handful of cards | Counts nearly full strength |

The weight is a homemade IDF, `1 / (1 + log(count))`. The percent shown on results is the raw similarity; the weight only affects ordering.

Name search runs on the pg_trgm extension: exact match, then prefix, then substring, then trigram similarity so "lightnig bolt" still finds Lightning Bolt. The autocomplete dropdown works the same way.

## Running locally

You need a Postgres with pgvector. Easiest way is docker:

```
docker run -d --name cardalike-db -e POSTGRES_PASSWORD=cards -p 5432:5432 pgvector/pgvector:pg16
```

Then seed it (first run embeds everything, give it a few minutes):

```
set DATABASE_URL=postgresql://postgres:cards@localhost:5432/postgres
pip install -r ingest/requirements.txt
python -m ingest.update
```

The schema applies itself on the first run, there is no separate setup step. Then start the site:

```
cd web
pip install -r requirements.txt
python app.py
```

## Deploying

The site runs on Railway and the updates run on GitHub Actions. One time setup:

1. On Railway, create a Postgres database (their pgvector template, or any Postgres image with pgvector). Copy the public connection string.
2. On GitHub, add that string as a repo secret named `DATABASE_URL` (Settings > Secrets and variables > Actions).
3. On the Actions tab, run the "update card data" workflow by hand once. That is the initial seed, takes a few minutes.
4. On Railway, create a service from this repo, set its root directory to `web/`, and give it the same `DATABASE_URL` as an environment variable. The Procfile handles the start command.

After that the workflow wakes up daily at 9am UTC and keeps the card data fresh on its own.

## The old version

The original file based version (`app.py`, `build_index.py`, root `templates/` and `static/`) still sits at the repo root so nothing breaks while the new stack gets deployed. Once the Railway site is up and verified, those can all be deleted along with the root `requirements.txt`.

## Tech stack

- **Backend:** Python / Flask
- **Database:** Postgres + pgvector on Railway
- **Frontend:** Jinja templates + vanilla JavaScript
- **Similarity:** sentence-transformers (`all-MiniLM-L6-v2`), embedded at ingest time only
- **Card data:** Scryfall bulk data (Oracle Cards), refreshed daily by GitHub Actions

## A typical search

1. Type a card name into the search bar.
2. The card appears with its image and full rules text.
3. Below it, the 20 closest cards show up, each with the line of text that matched and how close it was.
4. Hover a matched line to see which of your card's lines it paired with.
5. Load 20 more keeps digging deeper into the rankings, and clicking any card opens it on Scryfall.
