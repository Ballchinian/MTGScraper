# Delvefall

A web app that finds Magic: The Gathering cards similar to the card you search for. The point is convenience: you type a card name and that is the whole cost, no query language to learn and no vocabulary to go and look up first. Under it, the matching is real work rather than word overlap: not cards that share words, and not "cards that go in the same deck", but cards whose abilities mean the same thing even when the wording is completely different, by comparing sentence embeddings of every line of rules text ever printed.

## Features

- Card search by name with forgiving matching
- Similarity results ranked by best matching line of rules text
- Every result shows the exact line that matched and a similarity percent
- Line picker: click any of the searched card's rules lines to search just that ability (or combine several), and the URL stays shareable
- Every result is scored twice and the two are averaged: once on rules text, once on concepts (the community tags the card carries). There used to be a slider to weight one against the other. It is gone, because testing settled the question: an even split beat every other setting including both ends, so the control was asking something with one right answer. The badge on each card is the average, the list is always ordered by the number you can see, and hovering it breaks that number back into its two halves
- Tag picker: the concepts half shows the community tags it is searching, and clicking one switches it off, so a card that happens to be tagged for four different things can be searched for the one you care about. Switching a tag off takes the tags it implied with it, and the score is renormalized over what's left, so the percent keeps meaning what it always meant. **Picking a line narrows the tags to that one ability**, which is the thing neither Scryfall nor Tagger can do, because both are card-level: a card tagged three ways gives you no way to know which tag belongs to which ability. The narrowing is inferred rather than given, so it is a guess and the page says so, faded tags can be clicked back on, and a wrong one can be reported
- Filters: colour identity in three readings (at most these colours like deckbuilding, exactly these, or including these), price range, mana value range, card types (pick several, any of them matches, so an Artifact Creature answers to both), commanders only (legendary creatures, front face), hide game changers, and cards that aren't commander legal stay hidden unless you tick "include illegal"
- The filters panel only shows a narrowing filter once you add it, and disables the inputs of one you remove, so anything visible in there is something being applied and nothing can filter from off screen. The count next to the word Filters answers "is anything narrowing this?" without opening it. An inverted range is reported by the browser's own validation bubble, which also opens the panel so the offending box can be fixed on the spot
- Results below 70% match wait behind a button at the end of the list, which opens them ten percentage points at a time. It says how much worse each step gets in words ("weaker matches", then "weaker still") rather than in percentages, because a published range reads as a bug when empty bands make it jump about. There is no knob for it: a control for "how similar" on a site about similarity was bloat, and the banding is what keeps the sorts meaningful, since sorting one band by price asks "the cheapest card that matches this well" rather than "the cheapest card with any resemblance at all". Changing the sort keeps whatever depth you had opened, since it is the same search reordered; applying a filter or refreshing returns you to the strong matches
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
- So are bad tags. Once a line is picked, "a tag on the wrong line?" reports a tag the picker set aside that the line really is about, or one it kept that the line is not. Which way the complaint runs is never asked, it is read off the attribution, because that answer is already known and making the user get a second thing right is how reports end up wrong. Those become the labelled cards the line picker itself is graded against
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

Every cleaned line goes through a fine tuned EmbeddingGemma model trained specifically on Magic rules text, which turns text into a normalized vector of 768 numbers. Unlike any off the shelf model it knows that "draw a card, then discard a card" and "discard a card: draw a card" are different things, which is what lets "you may draw a card unless that player pays {4}" match "they may pay {1}. If the player does, they draw a card".

The current model was trained on a different question than the first one. The original learned "these two lines mean the same thing", which is the right target for ranking and the wrong one for saying what a line is ABOUT. The one running now learns to put a line next to the words of the tags it carries, trained on the eleven thousand cards whose entire rules text is a single line, where every tag a human typed belongs to that one line with no inference needed. That is a large amount of supervision nobody had to label. It moved the tag half of the site from 47% to 78% on the exam that matters, took line attribution from 88% to 94% precision, and held its ground on the older line-to-line test, which is the one that would have caught it forgetting what it already knew.

### Ranking

A search grabs the card's own lines from the database, runs a pgvector nearest neighbor query for each one (`<=>` is cosine distance, and the vectors are normalized so `1 - distance` is the real similarity), and keeps each candidate card's best matching pair. Common lines get weighted down so they don't drown out the interesting matches:

| Line | Rough count | Effect on ranking |
| --- | --- | --- |
| "Flying" | Thousands of cards | Heavily downweighted |
| A wordy triggered ability | A handful of cards | Counts nearly full strength |

The weight is a homemade IDF that leaves lines on 5 or fewer cards at full strength (a line shared by 2 cards is a functional reprint, exactly the match people came for), then falls off gently: `1 / (1 + log10(count / 5))`. Lines are counted by shape rather than by exact text, mana symbols collapsed to a placeholder, because a keyword whose cost varies otherwise splits into one and two card texts that each look unique: Overload is 27 card-lines across 22 different printed costs, so every one of them drew full weight and Vandalblast matched Dynacharge at 99% on the keyword and nothing else. The percent shown on results is a calibrated display score, pinned to hand-judged pairs so 80 marks the real quality boundary; the weight only affects ordering.

Results split around a 70% cutoff, which is where a blended score sits: the badge is the average of two axes, and averages land lower than either half. The strong tier is what you see; everything under it is filed into 10 point bands, and when a tier runs out the load more button offers the next band down by count and by how much worse it is in words, under its own labelled divider. The percent range is deliberately not published: skipped empty bands make it jump about, so it reads as a bug even when it is right. Empty bands are skipped rather than offered and found empty. The depth you have reached survives exactly one thing, a sort change, because reordering is not researching: the same cards come back in the new order. It rides sessionStorage for that one hop and is consumed on the way in, so a plain refresh does not restore it either. Applying a filter or starting a new search puts you back at the strong matches rather than pouring the old depth of weak results over a search you just changed.

One promise holds all of this together: the number on the badge is the number the cutoff uses. Nothing under the line ever appears above the fold, and the list always reads in descending order of the figure you can actually see. That is also why picking a line with no tags on it, a bare "Vigilance, trample, haste", drops the concepts half entirely rather than scoring it zero and halving everything: a perfect textual match would otherwise badge 50% while the gate let it through on 100. Weak matches can never leapfrog strong ones, and one band can never leapfrog another, which is what makes the sorts survive down here: sorting the whole tail by price returned the cheapest 0% match in the database, an answer to a question nobody asked. Filters run inside the nearest neighbor query itself, so a narrow search digs deeper into the rankings instead of thinning out an already-fetched list. Sorting by price happens after all of that: filter by relevance, sort by whatever you like, ties broken by match score. Price never mixes into the similarity percent itself, so the number always means one thing.

Name search runs on the pg_trgm extension: exact match, then prefix, then substring, then trigram similarity so "lightnig bolt" still finds Lightning Bolt. The autocomplete dropdown works the same way.

### The feedback loop

The results page can report three things: a card that shouldn't be in the results (the flag under each card, which asks the user to say why in their own words, nobody is expected to name a better card off the top of their head), a card that should have been there but isn't (the link in the results heading, which asks for the card's name), and a tag on the wrong line (the link under the tag picker, once a line is picked). Missing card reports get diagnosed before anything is stored: when a filter is what's hiding the card, the reporter is told exactly which one on the spot and the report never enters the queue, because that isn't the model's fault. Real gaps land in the `feedback` table with the reason, the scores and the model version at report time.

Tag reports never ask which direction the complaint runs. A tag is either on the picked line right now or it is not, and that decides whether the user is saying "you set this aside and the line IS about it" or "you kept this and the line is NOT". The answer is already in the database, and the direction is recorded at report time rather than looked up later, because the attribution gets rebuilt nightly and the report has to still make sense afterwards.

A review page at `/admin?key=...` (it only exists when the `ADMIN_KEY` environment variable is set on the web service) shows pending reports with the cards side by side, accept/reject buttons, and exports accepted ones in the format of whichever eval file they belong to: misplaced reports become triplet negatives with the match left to fill in by hand, missing reports become pair entries, and tag reports become labelled lines for the attribution exam, which tests the one thing the others cannot, whether a tag landed on the right line rather than on the right card. Those files are the hand checked test sets the next model is graded against, so user complaints literally turn into the exam.

## Tech stack

- **Backend:** Python / Flask
- **Database:** Postgres + pgvector on Railway
- **Frontend:** Jinja templates + vanilla JavaScript
- **Similarity:** sentence-transformers (a fine-tuned EmbeddingGemma), embedded at ingest time only
- **Card data:** Scryfall bulk data (Oracle Cards), refreshed daily by GitHub Actions

## A typical search

1. Type a card name into the search bar.
2. The card appears with its image and full rules text.
3. Below it, the 20 closest cards show up, each with the line of text that matched, its price and how close the match was.
4. Only care about one of the card's abilities? Click that line and the search reruns on just it, on both halves of the score: the tags narrow to the ones that ability is about, and the rest fade out of the way. Click more lines to combine them.
5. Narrow things down with the filter bar: colors your deck can play, a price budget, the card types you want, no game changers, commander legal only. Sort by price when you are hunting a cheaper version of something, and watch the arrow next to each price for which way it moves against the card you searched.
6. Hover a matched line to see which of your card's lines it paired with, and "+2 more matching lines" means the card matched more than one ability.
7. Load 20 more keeps digging deeper into the rankings, and clicking any card opens it on Scryfall.