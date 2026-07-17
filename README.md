# Delvefall

A web app that finds Magic: The Gathering cards that do similar things to the card you search for. Not cards that share words, and not "cards that go in the same deck", but cards whose abilities mean the same thing even when the wording is completely different, all by comparing sentence embeddings of every line of rules text ever printed.

## Features

- Card search by name with forgiving matching
- Similarity results ranked by best matching line of rules text
- Every result shows the exact line that matched and a similarity percent
- Line picker: click any of the searched card's rules lines to search just that ability (or combine several), and the URL stays shareable
- Filters: color identity (fits-within, like deckbuilding), price range, mana value range, card type, commanders only (legendary creatures), hide game changers, and cards that aren't commander legal stay hidden unless you tick "include illegal"
- Results below 80% match (adjustable in the filter bar) wait behind a "show weaker matches" button at the end of the list, so weak coincidences never crowd out real matches but nothing is unreachable
- Sort by best match or by price, in dollars or euros, with prices refreshed from Scryfall daily
- Results that match several of your card's lines say so ("+2 more matching lines")
- Load more button that pulls the next 20 results without a page reload
- Scryfall styled interface with card images linking back to Scryfall
- A landing page with the recent cards you searched floating around the search bar, click one to run it again
- Unique cards: the counterpart to a random card page. It deals one random card nothing else in the game resembles the moment you arrive, with a repeat button, an adjustable uniqueness bar, the same filters as search, and a per-device memory so you never get dealt the same card twice (until you start over)
- The dealt cards form a trail: back/forward arrows revisit everything you've ever been dealt
- Every card picture on the site (the searched card, the results grid, the unique page) gets hover buttons where the physical card needs them: sideways cards (battles, split cards) stay vertical so the grid reads uniform, with rotate to turn them readable, Kamigawa flip cards flip 180, double faced cards transform. Invasion of Zendikar does all of it at once
- Updates itself! A bot checks Scryfall every day and new cards just appear
- Bad results are reportable: every result carries a quiet "shouldn't be here?" flag and the results heading an "expected a card that isn't here?" link. Reports about filters get answered on the spot, real matching gaps become test cases for the next model

## How it works

Everything lives in a Postgres database with the pgvector extension. The embeddings sit in a `vector(768)` column and the database does the nearest neighbor math itself, so the website is a tiny Flask app that runs a few queries per search. The language model only ever runs inside the update pipeline, never on the server.

Three pieces:

| Piece | What it does |
| --- | --- |
| `web/` | The Flask site. Talks to Postgres, needs no torch, no giant files, barely any memory |
| `ingest/` | The update pipeline. Downloads Scryfall's bulk data, embeds whatever is new or changed, writes it to the database |
| `.github/workflows/update.yml` | Runs the pipeline every day on GitHub Actions so the heavy dependencies never go anywhere near the web server |

`common/` holds the bits both sides share: the database schema and the card cleaning helpers.

### The database

Five tables. `cards` has one row per unique card, keyed by Scryfall's `oracle_id` (stable across every printing), including the filter columns: color identity, USD and EUR prices, mana value, the official Commander game changer flag and commander legality. `lines` has one row per line of rules text with its embedding. `line_stats` counts how many cards share each exact line, for the ranking weights. `meta` remembers which Scryfall bulk file was processed last and which model made the vectors. `feedback` holds user reports from the results page (see the feedback loop below).

Two derived columns power the unique cards page: `lines.nn_sim` is each line's nearest neighbor similarity (how close the closest line on any *other* card gets), and `cards.uniqueness` rolls that up as 1 minus the card's most isolated line's `nn_sim`, so a card with Flying plus one ability nobody else has still counts as unique, because uniqueness is judged per line, not by the card's best match. The ingest recomputes them whenever lines change, from scratch rather than incrementally: a new card can make an old card less unique and a deleted card can make its old neighbors more unique, so patching only changed rows would quietly rot the scores. The all-pairs math runs as one big numpy matrix multiply on the GitHub Actions runner (about a minute) instead of ~31k pgvector scans (hours of busy production database). All-pairs work belongs next to the big CPU, one-query-at-a-time work belongs next to the data.

Prices show in dollars or euros: a switch in the extra filters panel flips every price on the page, and the price bounds and price sorts follow it, so what you filter on is always the number you see. Prices are the cheapest paper printing in any finish, found by streaming Scryfall's Default Cards file (every printing, a couple of gigabytes) through ijson each day. Digital printings, oversized promos and gold border world championship decks don't count, you can't sleeve those up.

There is deliberately no vector index: at ~61k rows Postgres scans everything in a few milliseconds and the results are exact, identical to the old in-memory version. If the game ever grows 10x there is a commented out HNSW index in `common/schema.sql` ready to go.

### Daily updates

The pipeline is built so that doing nothing costs nothing:

1. Ask Scryfall's bulk data API for the Oracle Cards file's `updated_at` timestamp (one tiny request).
2. If it matches the one stored in `meta`, stop. Done in two seconds.
3. Otherwise download the bulk file and hash every card's name + rules text. Cards whose hash matches the database get skipped without embedding anything.
4. The Default Cards file gets streamed through for everything's cheapest printing, and every card's row gets rewritten regardless, because prices move daily and the game changer list gets edited, and none of that shows up in the text hash.
5. Only genuinely new or changed cards (usually a handful, or zero) get their lines embedded and written, all inside one transaction.
6. The line counts get rebuilt and the new timestamp saved.
7. If any lines changed, the uniqueness scores get recomputed (see the database section for why that is always a full recompute).

Running the same pipeline against an empty database seeds the whole thing, everything counts as new and ~61k lines get embedded in one big batch. That is the entire initial setup.

Cards that disappear from the bulk file, or that the filters newly exclude, get deleted at the end of the run. That way tightening a filter cleans the database up on its own.

### Card data

Card data comes from Scryfall's bulk data API, which publishes a daily "Oracle Cards" file with exactly one entry per unique card (Scryfall asks tools to use this instead of scraping pages). Requests send a custom User Agent like their docs require. Cards are filtered before indexing:

| Filter | Why |
| --- | --- |
| Joke sets (funny / memorabilia) | Not real cards |
| Tokens, emblems, art cards, schemes | Not playable cards |
| Digital only (Alchemy, MTGO exclusives) | Never printed in paper |
| No rules text | Vanilla creatures and basic lands have nothing to compare |

### Text cleaning

Each line of a card's rules text is treated as one ability, so lines get embedded separately and one matching ability is enough. Before embedding, every line is cleaned: reminder text in parentheses is stripped, and the card's references to its own name are swapped for "this card" so names can't influence matching (including the shortened first name that legendary cards use mid sentence).

### Embeddings

Every cleaned line goes through a fine tuned EmbeddingGemma model trained specifically on Magic rules text (the whole story of building it lives in `finetune/README.md`), which turns text into a normalized vector of 768 numbers where lines that mean similar things land close together, and unlike any off the shelf model, knows that "draw a card, then discard a card" and "discard a card: draw a card" are different things. That is what lets "you may draw a card unless that player pays {4}" match "they may pay {1}. If the player does, they draw a card".

### Ranking

A search grabs the card's own lines from the database, runs a pgvector nearest neighbor query for each one (`<=>` is cosine distance, and the vectors are normalized so `1 - distance` is the real similarity), and keeps each candidate card's best matching pair. Common lines get weighted down so they don't drown out the interesting matches:

| Line | Rough count | Effect on ranking |
| --- | --- | --- |
| "Flying" | Thousands of cards | Heavily downweighted |
| A wordy triggered ability | A handful of cards | Counts nearly full strength |

The weight is a homemade IDF that leaves lines on 5 or fewer cards at full strength (a line shared by 2 cards is a functional reprint, exactly the match people came for), then falls off gently: `1 / (1 + log10(count / 5))`. The percent shown on results is a calibrated display score, pinned to hand-judged pairs so 80 marks the real quality boundary; the weight only affects ordering.

Results split into two tiers around a minimum match percent (80 by default, relaxing to 70 when the slider blends both axes, adjustable in the filter bar). The strong tier is what you see; when it runs out, the load more button turns into "show weaker matches" with a count and keeps paging through the weak tier below a divider. Weak matches can never leapfrog strong ones, even when sorting by price, which is exactly when a cheap 50% coincidence would otherwise sit on top. Filters run inside the nearest neighbor query itself, so a narrow search digs deeper into the rankings instead of thinning out an already-fetched list. Sorting by price happens after all of that: filter by relevance, sort by whatever you like, ties broken by match score. Price never mixes into the similarity percent itself, so the number always means one thing.

Name search runs on the pg_trgm extension: exact match, then prefix, then substring, then trigram similarity so "lightnig bolt" still finds Lightning Bolt. The autocomplete dropdown works the same way.

### The feedback loop

The results page can report two things: a card that shouldn't be in the results (the flag under each card, which asks the user to say why in their own words, nobody is expected to name a better card off the top of their head) and a card that should have been there but isn't (the link in the results heading, which asks for the card's name). Missing card reports get diagnosed before anything is stored: when a filter is what's hiding the card, the reporter is told exactly which one on the spot and the report never enters the queue, because that isn't the model's fault. Real gaps land in the `feedback` table with the reason, the scores and the model version at report time.

A review page at `/admin?key=...` (it only exists when the `ADMIN_KEY` environment variable is set on the web service) shows pending reports with the cards side by side, accept/reject buttons, and exports accepted ones in the markdown format of the eval files: misplaced reports become `finetune/triplets.md` negatives with the match left to fill in by hand, missing reports become `finetune/pairs.md` entries. Those files are the hand checked test sets the next model is graded against, so user complaints literally turn into the exam.

## Tech stack

- **Backend:** Python / Flask
- **Database:** Postgres + pgvector on Railway
- **Frontend:** Jinja templates + vanilla JavaScript
- **Similarity:** sentence-transformers (a fine-tuned EmbeddingGemma, see `finetune/README.md`), embedded at ingest time only
- **Card data:** Scryfall bulk data (Oracle Cards), refreshed daily by GitHub Actions

## A typical search

1. Type a card name into the search bar.
2. The card appears with its image and full rules text.
3. Below it, the 20 closest cards show up, each with the line of text that matched, its price and how close the match was.
4. Only care about one of the card's abilities? Click that line and the search reruns on just it. Click more lines to combine them.
5. Narrow things down with the filter bar: colors your deck can play, a price budget, a card type, no game changers, commander legal only. Sort by price when you are hunting a cheaper version of something.
6. Hover a matched line to see which of your card's lines it paired with, and "+2 more matching lines" means the card matched more than one ability.
7. Load 20 more keeps digging deeper into the rankings, and clicking any card opens it on Scryfall.
