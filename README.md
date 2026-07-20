# Delvefall

A web app that finds Magic: The Gathering cards similar to the card you search for. The point is convenience: you type a card name and that is the whole cost, no query language to learn and no vocabulary to go and look up first. Under it, the matching is real work rather than word overlap: not cards that share words, and not "cards that go in the same deck", but cards whose abilities mean the same thing even when the wording is completely different, by comparing sentence embeddings of every line of rules text ever printed.

## Features

- Card search by name with forgiving matching
- Similarity results ranked by best matching line of rules text
- Every result shows the exact line that matched and a similarity percent
- Line picker: click any of the searched card's rules lines to search just that ability (or combine several), and the URL stays shareable
- Tag picker: the concepts side of the slider shows the community tags it is searching, and clicking one switches it off, so a card that happens to be tagged for four different things can be searched for the one you care about. Switching a tag off takes the tags it implied with it, and the score is renormalized over what's left, so the percent keeps meaning what it always meant. Picking a line narrows the tags to that one ability. **Dark for now:** the whole picker only appears where the `LINE_TAGS` environment variable is set, because the line-to-tag attribution that makes it worth having reads 88% precision and 82% recall, and a picked line still quietly sets aside tags it shouldn't. It ships when that clears 95%. The slider is unaffected either way, the concepts side still scores off the whole card's tags
- Filters: colour identity in three readings (at most these colours like deckbuilding, exactly these, or including these), price range, mana value range, card types (pick several, any of them matches, so an Artifact Creature answers to both), commanders only (legendary creatures, front face), hide game changers, and cards that aren't commander legal stay hidden unless you tick "include illegal"
- The filters panel only shows a narrowing filter once you add it, and disables the inputs of one you remove, so anything visible in there is something being applied and nothing can filter from off screen. The count next to the word Filters answers "is anything narrowing this?" without opening it. An inverted range is reported by the browser's own validation bubble, which also opens the panel so the offending box can be fixed on the spot
- Results below 80% match (70 where the slider blends both axes) wait behind a button at the end of the list, which opens them ten percentage points at a time. It says how much worse each step gets in words ("weaker matches", then "weaker still") rather than in percentages, because the cutoff moves with the slider and a range would mean something different at each position. There is no knob for it: a control for "how similar" on a site about similarity was bloat, and the banding is what keeps the sorts meaningful, since sorting one band by price asks "the cheapest card that matches this well" rather than "the cheapest card with any resemblance at all". Applying a filter or reloading returns you to the strong matches, since changing a filter is the moment you want to see what the new search found
- Sort by best match, by price (in dollars, euros or approximate pounds, refreshed from Scryfall daily), by how much a card actually gets played (most or least, from EDHREC rank), or by release date, newest or oldest first
- Results that match several of your card's lines say so ("+2 more matching lines")
- Load more button that pulls the next 20 results without a page reload
- An ink and paper skin with the filter widgets drawn by hand and the real mana symbols on the color checkboxes and cost lines (Scryfall's SVGs, self hosted), and every card image links back to Scryfall
- A landing page with the recent cards you searched floating around the search bar, click one to run it again, and before you have a history much-played cards fill the slots so the room is never empty
- Unique cards: the counterpart to a random card page. It deals one card nothing else in the game resembles the moment you arrive, with a repeat button, the same filters as search, and a per-device memory so you never get dealt the same card twice (until you start over). There is no uniqueness bar to set: each deal draws at random from the cards within a small band of the most unique one still unseen, so it stays near the top of what is left while giving two people with the same filters different runs
- The dealt cards form a trail: back/forward arrows revisit everything you've ever been dealt
- Every card picture on the site (the searched card, the results grid, the unique page) gets hover buttons where the physical card needs them: sideways cards (battles, split cards) stay vertical so the grid reads uniform, with rotate to turn them readable, Kamigawa flip cards flip 180, double faced cards transform. Invasion of Zendikar does all of it at once
- Updates itself! A bot checks Scryfall every day and new cards just appear
- Bad results are reportable: every result carries a quiet "shouldn't be here?" flag and the results heading an "expected a card that isn't here?" link. Reports about filters get answered on the spot, real matching gaps become test cases for the next model
- Links unfurl: every page carries Open Graph tags, so pasting a card's results into Discord shows the card's image and what the page is. Each card has one canonical URL no matter which filters found it, and a sitemap plus breadcrumbs hand search engines the whole card pool without making them walk result pages

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

Five tables. `cards` has one row per unique card, keyed by Scryfall's `oracle_id` (stable across every printing), including the filter columns: color identity, USD and EUR prices, mana value, the official Commander game changer flag and commander legality, plus the EDHREC rank and first printing date that power the played and release date sorts. `lines` has one row per line of rules text with its embedding. `line_stats` counts how many cards share each line, for the ranking weights. `meta` remembers which Scryfall bulk file was processed last and which model made the vectors. `feedback` holds user reports from the results page (see the feedback loop below).

Two derived columns power the unique cards page: `lines.nn_sim` is each line's nearest neighbor similarity (how close the closest line on any *other* card gets), and `cards.uniqueness` rolls that up as 1 minus the card's most isolated line's `nn_sim`, so a card with Flying plus one ability nobody else has still counts as unique, because uniqueness is judged per line, not by the card's best match. The ingest recomputes them whenever lines change, from scratch rather than incrementally: a new card can make an old card less unique and a deleted card can make its old neighbors more unique, so patching only changed rows would quietly rot the scores. The all-pairs math runs as one big numpy matrix multiply on the GitHub Actions runner (about a minute) instead of ~31k pgvector scans (hours of busy production database). All-pairs work belongs next to the big CPU, one-query-at-a-time work belongs next to the data.

Prices show in dollars, euros or pounds: a switch on the price row of the filters panel flips every price on the page, and the price bounds and price sorts follow it, so what you filter on is always the number you see. Prices are the cheapest paper printing in any finish, found by streaming Scryfall's Default Cards file (every printing, a couple of gigabytes) through ijson each day. Digital printings, oversized promos and gold border world championship decks don't count, you can't sleeve those up.

Pounds are derived rather than sourced, and labelled approximate because of it: Scryfall quotes dollars and euros only, so the pound figure converts both at the day's ECB reference rates and takes the middle, falling back to whichever one a card actually has. The rate fetch is cached for a day and seeded with a fallback pair, so an outage at the rate API can never break a page.

The nearest neighbor scans walk an HNSW index, which turned 200-250ms of vector math per line into about 20ms. The build is deliberately denser than pgvector's defaults (m=32, ef_construction=200): common lines put hundreds of identical embeddings into the graph, the default build leaves those clusters badly connected, and a 94% match once fell out of a top-400 scan because of it. The dense build measured zero misses above 0.90 similarity against the exact scan. Searches use pgvector's iterative scan in strict order, so a heavily filtered search keeps walking the graph until it has real answers instead of coming back short.

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

The weight is a homemade IDF that leaves lines on 5 or fewer cards at full strength (a line shared by 2 cards is a functional reprint, exactly the match people came for), then falls off gently: `1 / (1 + log10(count / 5))`. Lines are counted by shape rather than by exact text, mana symbols collapsed to a placeholder, because a keyword whose cost varies otherwise splits into one and two card texts that each look unique: Overload is 27 card-lines across 22 different printed costs, so every one of them drew full weight and Vandalblast matched Dynacharge at 99% on the keyword and nothing else. The percent shown on results is a calibrated display score, pinned to hand-judged pairs so 80 marks the real quality boundary; the weight only affects ordering.

Results split around a cutoff the slider sets (80 at either end, 70 on the mixed detents, where a blended score is an average and averages land lower). The strong tier is what you see; everything under it is filed into 10 point bands, and when a tier runs out the load more button offers the next band down by count and by how much worse it is in words, under its own labelled divider. The percent range is deliberately not published: it moves with the slider's cutoff and skipped empty bands make it jump about, so it reads as a bug even when it is right. Empty bands are skipped rather than offered and found empty. The depth you have reached lives in a page variable and nowhere else, deliberately: applying a filter or reloading puts you back at the strong matches rather than pouring the old depth of weak results over a search you just changed. Weak matches can never leapfrog strong ones, and one band can never leapfrog another, which is what makes the sorts survive down here: sorting the whole tail by price returned the cheapest 0% match in the database, an answer to a question nobody asked. Filters run inside the nearest neighbor query itself, so a narrow search digs deeper into the rankings instead of thinning out an already-fetched list. Sorting by price happens after all of that: filter by relevance, sort by whatever you like, ties broken by match score. Price never mixes into the similarity percent itself, so the number always means one thing.

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
5. Narrow things down with the filter bar: colors your deck can play, a price budget, the card types you want, no game changers, commander legal only. Sort by price when you are hunting a cheaper version of something, and watch the arrow next to each price for which way it moves against the card you searched.
6. Hover a matched line to see which of your card's lines it paired with, and "+2 more matching lines" means the card matched more than one ability.
7. Load 20 more keeps digging deeper into the rankings, and clicking any card opens it on Scryfall.