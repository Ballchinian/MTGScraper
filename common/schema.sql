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
    price_usd       numeric,
    price_eur       numeric,  --stored for a future currency toggle, the site only shows usd
    cmc             numeric NOT NULL DEFAULT 0,  --mana value. numeric because scryfall says so, in practice whole numbers
    game_changer    boolean NOT NULL DEFAULT false,
    legal_commander boolean NOT NULL DEFAULT true
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

--trigram index so the name searches (prefix, substring, fuzzy) stay quick
CREATE INDEX IF NOT EXISTS cards_name_trgm ON cards USING gin (name gin_trgm_ops);

--one row per line of rules text, with its embedding (384 numbers from
--all-MiniLM-L6-v2, normalized, so cosine distance works)
CREATE TABLE IF NOT EXISTS lines (
    id        bigserial PRIMARY KEY,
    oracle_id uuid NOT NULL REFERENCES cards(oracle_id) ON DELETE CASCADE,
    line_text text NOT NULL,
    embedding vector(384) NOT NULL
);

--lets us grab one card's lines instantly at search time
CREATE INDEX IF NOT EXISTS lines_oracle_id ON lines (oracle_id);

--no vector index on purpose! at ~61k rows postgres scans them all in a few
--milliseconds and the results are exact, identical to the old numpy version.
--if the table ever grows 10x, uncomment this for approximate-but-fast search:
--CREATE INDEX lines_embedding_hnsw ON lines USING hnsw (embedding vector_cosine_ops);

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
