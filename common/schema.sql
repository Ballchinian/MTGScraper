--the whole database in one file. everything is IF NOT EXISTS so its safe to
--run over and over. ingest/update.py runs this at the start of every run,
--which means a brand new empty database sets itself up on the first run

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

--one row per unique card. scryfall's oracle_id stays the same across every
--printing of a card, so its the perfect primary key. text_hash is how the
--updater spots cards whose text changed without comparing whole strings
CREATE TABLE IF NOT EXISTS cards (
    oracle_id    uuid PRIMARY KEY,
    name         text NOT NULL,
    mana_cost    text,
    type_line    text,
    oracle_text  text,
    image        text,
    scryfall_uri text,
    text_hash    text NOT NULL,
    updated_at   timestamptz DEFAULT now(),
    --the filter columns. the ingest refreshes these on every run even when
    --the rules text didnt change, since prices move every day
    color_identity  text NOT NULL DEFAULT '',
    price_usd       numeric,  --the cheapest paper printing in any finish, not scryfall's preferred printing
    price_eur       numeric,  --the cheapest paper printing in euros, the currency toggle's other half
    cmc             numeric NOT NULL DEFAULT 0,  --mana value. numeric because scryfall says so, in practice whole numbers
    game_changer    boolean NOT NULL DEFAULT false,
    legal_commander boolean NOT NULL DEFAULT true,
    layout          text NOT NULL DEFAULT 'normal',
    image_back      text NOT NULL DEFAULT ''
);

--databases created before the filter columns existed pick them up here.
--fresh ones already have them from the CREATE TABLE above, and IF NOT
--EXISTS makes rerunning free either way
ALTER TABLE cards ADD COLUMN IF NOT EXISTS color_identity text NOT NULL DEFAULT '';
ALTER TABLE cards ADD COLUMN IF NOT EXISTS price_usd numeric;
ALTER TABLE cards ADD COLUMN IF NOT EXISTS price_eur numeric;
ALTER TABLE cards ADD COLUMN IF NOT EXISTS cmc numeric NOT NULL DEFAULT 0;
ALTER TABLE cards ADD COLUMN IF NOT EXISTS game_changer boolean NOT NULL DEFAULT false;
ALTER TABLE cards ADD COLUMN IF NOT EXISTS legal_commander boolean NOT NULL DEFAULT true;

--the /unique page. a card's uniqueness is its most isolated line: 1 minus
--the best match that line has anywhere else in the game. its judged per
--line on purpose, a card with Flying plus one ability nobody else has IS
--unique in the "could define a deck" sense, even though the Flying line
--matches thousands of cards. unique_line remembers which line earned the
--score so the page can show it. both stay NULL for cards with no
--searchable lines, which quietly keeps them out of the unique deck
ALTER TABLE cards ADD COLUMN IF NOT EXISTS uniqueness real;
ALTER TABLE cards ADD COLUMN IF NOT EXISTS unique_line text;

--the concept counterpart: how alone a card is in tag space, 1 minus the
--best cosine any other card's tag vector manages. filled by ingest/tags.py
--so /unique can slide between rules-text-unique and concept-unique. stays
--NULL for untagged cards, unknown is not the same as unique
ALTER TABLE cards ADD COLUMN IF NOT EXISTS concept_uniqueness real;

--scryfall's edhrec popularity rank, 1 is the most played card in the
--format. powers the most/least played sorts. NULL means unranked, which
--reads as maximally obscure
ALTER TABLE cards ADD COLUMN IF NOT EXISTS edhrec_rank int;

--how the card physically works, straight from scryfall: 'split' and battle
--type lines mean the picture is printed sideways and the site offers a
--rotate button, 'flip' means the bottom half reads upside down, and
--image_back holds the other face's picture when one exists so the card can
--be turned over on the page
ALTER TABLE cards ADD COLUMN IF NOT EXISTS layout text NOT NULL DEFAULT 'normal';
ALTER TABLE cards ADD COLUMN IF NOT EXISTS image_back text NOT NULL DEFAULT '';

--trigram index so the name searches (prefix, substring, fuzzy) stay quick
CREATE INDEX IF NOT EXISTS cards_name_trgm ON cards USING gin (name gin_trgm_ops);

--one row per line of rules text, with its embedding (768 numbers from my
--fine tuned embeddinggemma, normalized, so cosine distance works). databases
--still on the old 384 column get moved over by update.py when it notices
--the model changed.
--
--nn_sim is the line's nearest neighbor similarity: how close the closest
--line on any OTHER card gets to this one. 1.0 means some other card has
--this exact ability, low means nothing else in the game does anything like
--it. update.py fills it in after the embeddings, its search turned inside
--out (search asks whats closest, this asks how far away even the closest
--thing is)
CREATE TABLE IF NOT EXISTS lines (
    id        bigserial PRIMARY KEY,
    oracle_id uuid NOT NULL REFERENCES cards(oracle_id) ON DELETE CASCADE,
    line_text text NOT NULL,
    embedding vector(768) NOT NULL,
    nn_sim    real,
    face      smallint NOT NULL DEFAULT 0
);

ALTER TABLE lines ADD COLUMN IF NOT EXISTS nn_sim real;

--which face printed the line, 0 front / 1 back. when the winning match
--lives on a card's back face the results page shows that side first, so
--the line under the picture is on the picture (the ulvenwald lesson: the
--back face really does print "{T}: Add {C}{C}.", the display just hid it)
ALTER TABLE lines ADD COLUMN IF NOT EXISTS face smallint NOT NULL DEFAULT 0;

--whole-card rows: one extra row per multi-line card holding its entire
--cleaned text, for the line-merging blind spot (two separate lines that
--together equal another card's compound line - shadrix vs gluntch). they
--are retrieval material for a future card-level scorer and stay OUT of
--everything line-shaped: uniqueness, line_stats, the per-line search and
--the training miner all filter on NOT whole
ALTER TABLE lines ADD COLUMN IF NOT EXISTS whole boolean NOT NULL DEFAULT false;

--lets us grab one card's lines instantly at search time
CREATE INDEX IF NOT EXISTS lines_oracle_id ON lines (oracle_id);

--approximate index for the search's nearest neighbor scans, ~20ms per line
--where the exact scan measured 200-250ms. the dense build parameters are
--load bearing: common lines put hundreds of identical embeddings in the
--graph, and the default m=16/ef_construction=64 build leaves those clusters
--badly connected (a 94% match at true rank 181 fell out of a top-400 scan).
--m=32/ef_construction=200 measured zero misses above 0.90 sim against the
--exact scan. partial on NOT whole to mirror the search's filter, so
--whole-card rows never enter the graph. scan settings live in web/db.py,
--uniqueness is unaffected, recompute_uniqueness does its math in numpy
CREATE INDEX IF NOT EXISTS lines_embedding_hnsw ON lines USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 200) WHERE (NOT whole);

--how many cards share each exact line of text, for the idf weighting
--("Flying" is on thousands of cards so it barely counts, a wordy triggered
--ability is nearly unique so it counts full strength)
CREATE TABLE IF NOT EXISTS line_stats (
    line_text text PRIMARY KEY,
    count     int NOT NULL
);

--little key/value table for bookkeeping. right now it only holds the
--scryfall bulk timestamp we processed last, which is what lets the daily
--update skip itself when theres nothing new
CREATE TABLE IF NOT EXISTS meta (
    key   text PRIMARY KEY,
    value text
);

--axis 2 (conceptual similarity) groundwork: community tags from scryfall
--tagger, via the official oracle_tags bulk file. one row per card-tag link.
--ingest/tags.py rebuilds both tables from scratch whenever the bulk file
--changes, same philosophy as line_stats
CREATE TABLE IF NOT EXISTS card_tags (
    oracle_id uuid NOT NULL REFERENCES cards(oracle_id) ON DELETE CASCADE,
    tag       text NOT NULL,
    PRIMARY KEY (oracle_id, tag)
);

CREATE INDEX IF NOT EXISTS card_tags_tag ON card_tags (tag);

--one row per tag that survived the trivia blocklist: its parents (tagger
--tags form a hierarchy, kept for rollup scoring later), how many of OUR
--cards carry it, the idf weight derived from that count (so broad tags like
--triggered-ability barely count), and the tagger description for tooltips
CREATE TABLE IF NOT EXISTS tags (
    tag         text PRIMARY KEY,
    parents     text[] NOT NULL DEFAULT '{}',
    card_count  int NOT NULL DEFAULT 0,
    idf         real NOT NULL DEFAULT 0,
    description text NOT NULL DEFAULT ''
);

ALTER TABLE tags ADD COLUMN IF NOT EXISTS idf real NOT NULL DEFAULT 0;

--derived at ingest, like line_stats: each card's idf-weighted tag vector
--length, so the concept query never recomputes 31k norms per search
CREATE TABLE IF NOT EXISTS card_tag_norms (
    oracle_id uuid PRIMARY KEY REFERENCES cards(oracle_id) ON DELETE CASCADE,
    norm      real NOT NULL
);

--user reports from the search page, the raw material for the next round of
--the eval files. kind 'missing' means "this good card should have been in
--the results" and carries expected_id (a future pairs.md entry), kind
--'misplaced' means "this bad card shouldnt be here" and carries got_id plus
--the user's reason in their own words (a future triplets.md negative).
--names are snapshotted alongside the ids on purpose: cards can vanish from
--the cards table between the report and the review, and a report thats lost
--its cards should still read. the percents and embed_model pin down what
--the site actually said at report time, since both move whenever the model
--changes. no foreign keys, same reason
CREATE TABLE IF NOT EXISTS feedback (
    id            bigserial PRIMARY KEY,
    kind          text NOT NULL,
    anchor_id     uuid NOT NULL,
    anchor_name   text NOT NULL,
    expected_id   uuid,
    expected_name text,
    got_id        uuid,
    got_name      text,
    expected_pct  int,
    got_pct       int,
    reason        text NOT NULL DEFAULT '',
    picked_lines  text NOT NULL DEFAULT '',
    filters       text NOT NULL DEFAULT '',
    embed_model   text NOT NULL DEFAULT '',
    ip            text NOT NULL DEFAULT '',
    status        text NOT NULL DEFAULT 'pending',
    created_at    timestamptz DEFAULT now()
);
