#the daily updater. asks scryfall if their bulk card file changed, and if it
#did, only does the heavy embedding work for cards that are actually new or
#have different text than last time. most days thats a handful of cards, so
#the run finishes in seconds after the download.
#
#running this against a brand new empty database seeds the whole thing
#(everything counts as new, ~61k lines to embed, takes a few minutes).
#github actions runs it every day, or run it yourself from the repo root:
#    python -m ingest.update
#with DATABASE_URL set to the postgres connection string

import os
import sys
import json
import time
import hashlib

import requests
import psycopg
from pgvector.psycopg import register_vector

from common.cards import HEADERS, keep_card, split_lines, get_text, get_image

BULK_URL = "https://api.scryfall.com/bulk-data"
DOWNLOAD_FILE = "oracle-cards.json"

#my fine tuned embeddinggemma (a sentence-transformers model). it sits in a
#private repo on hugging face, so HF_TOKEN has to be set or the download 401s.
#the prompt was glued to the front of every line during training, encoding
#without it gives useless vectors. swapping EMBED_MODEL for something else
#makes the next run rebuild every vector on its own
EMBED_MODEL = "BallchinianMan/mtg-tuned-embeddinggemma-300m"
EMBED_PROMPT = "task: sentence similarity | query: "
EMBED_DIMS = 768


def get_with_retries(url, tries=3):
    #scryfall hiccups sometimes, no reason to fail the whole run over it
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == tries - 1:
                raise
            wait = 5 * (attempt + 1)
            print("request failed (" + str(e) + "), retrying in " + str(wait) + "s...")
            time.sleep(wait)


def download_bulk(url):
    #stream the file straight to disk instead of holding 150mb of raw json in
    #memory on top of the parsed version
    print("downloading " + url)
    print("(its like 150mb so this can take a minute)")
    for attempt in range(3):
        try:
            with requests.get(url, headers=HEADERS, timeout=300, stream=True) as r:
                r.raise_for_status()
                with open(DOWNLOAD_FILE, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            return
        except Exception as e:
            if attempt == 2:
                raise
            print("download failed (" + str(e) + "), retrying...")
            time.sleep(10)


def card_hash(card):
    #the name is part of the hash on purpose: clean_line swaps the card's own
    #name for "this card" inside the text, so a renamed card needs its lines
    #rebuilt even if the text itself looks the same
    return hashlib.sha256((card["name"] + "\n" + get_text(card)).encode("utf-8")).hexdigest()


def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("set DATABASE_URL first (the postgres connection string)")
        sys.exit(1)

    conn = psycopg.connect(db_url)

    #make sure the tables exist. schema.sql is all IF NOT EXISTS so this is
    #free on every run after the first
    schema_path = os.path.join(os.path.dirname(__file__), "..", "common", "schema.sql")
    with open(schema_path, encoding="utf-8") as f:
        conn.execute(f.read())
    conn.commit()
    register_vector(conn)

    #vectors from two different models cant be compared with each other, so
    #if the database was embedded by anything other than the model above,
    #every line needs redoing this run
    row = conn.execute("SELECT value FROM meta WHERE key = 'embed_model'").fetchone()
    model_changed = row is None or row[0] != EMBED_MODEL
    if model_changed:
        print("embedding model changed, this run rebuilds every vector (the slow full reseed)")

    print("asking scryfall where the bulk file lives...")
    bulk = None
    for item in get_with_retries(BULK_URL).json()["data"]:
        #oracle_cards = one entry per unique card instead of every single printing
        if item["type"] == "oracle_cards":
            bulk = item
    updated_at = bulk["updated_at"]

    #the gate: if we already processed this exact bulk file, stop right here.
    #this is what makes rerunning the workflow basically free. unless the
    #model changed, then theres a full rebuild to do either way
    row = conn.execute("SELECT value FROM meta WHERE key = 'scryfall_updated_at'").fetchone()
    if not model_changed and row and row[0] == updated_at:
        print("already processed the bulk file from " + updated_at + ", nothing to do")
        conn.close()
        return

    download_bulk(bulk["download_uri"])
    print("loading cards...")
    with open(DOWNLOAD_FILE, encoding="utf-8") as f:
        all_cards = json.load(f)
    os.remove(DOWNLOAD_FILE)
    print("scryfall gave us " + str(len(all_cards)) + " cards")

    cards = []
    for c in all_cards:
        if keep_card(c):
            cards.append(c)
    print("kept " + str(len(cards)) + " real cards that have rules text")

    #what do we already have? oracle_id -> hash of the text we embedded last time
    have = {}
    for oracle_id, text_hash in conn.execute("SELECT oracle_id, text_hash FROM cards"):
        have[str(oracle_id)] = text_hash

    if model_changed:
        #forget the stored hashes so every card counts as new and gets
        #re-embedded. the old vectors stay put for now, the site keeps
        #searching on them while the new ones compute
        have = {}

    new_cards = []
    changed_cards = []
    unchanged = 0
    card_rows = []  #every kept card, for the upsert below
    for c in cards:
        h = card_hash(c)
        #prices come from scryfall as strings like "0.25" or None, postgres
        #is happy with either going into a numeric column
        prices = c.get("prices", {})
        card_rows.append((c["oracle_id"], c["name"], c.get("mana_cost", ""), c.get("type_line", ""),
                          get_text(c), get_image(c), c.get("scryfall_uri", ""), h,
                          "".join(c.get("color_identity", [])), prices.get("usd"), prices.get("eur"),
                          c.get("cmc", 0), c.get("game_changer", False),
                          c.get("legalities", {}).get("commander") == "legal"))
        old = have.get(c["oracle_id"])
        if old is None:
            new_cards.append((c, h))
        elif old != h:
            changed_cards.append((c, h))
        else:
            unchanged += 1

    #cards in the database that arent in the kept list anymore, either scryfall
    #dropped them or the filters got stricter. they need to go or they sit in
    #search results forever
    kept_ids = set()
    for c in cards:
        kept_ids.add(c["oracle_id"])
    stale = []
    for oid in have:
        if oid not in kept_ids:
            stale.append(oid)
    print(str(len(new_cards)) + " new, " + str(len(changed_cards)) + " changed, "
          + str(unchanged) + " unchanged, " + str(len(stale)) + " to remove")

    #every card row gets written every run, not just the new and changed ones.
    #prices move daily and wizards edits the game changer list now and then,
    #so waiting for a rules text change would leave those stale forever. the
    #slow embedding work below still only happens when text actually changed
    print("writing " + str(len(card_rows)) + " card rows (keeps prices and filters fresh)...")
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO cards (oracle_id, name, mana_cost, type_line, oracle_text, image, scryfall_uri, text_hash,
                               color_identity, price_usd, price_eur, cmc, game_changer, legal_commander, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (oracle_id) DO UPDATE SET
                name = EXCLUDED.name,
                mana_cost = EXCLUDED.mana_cost,
                type_line = EXCLUDED.type_line,
                oracle_text = EXCLUDED.oracle_text,
                image = EXCLUDED.image,
                scryfall_uri = EXCLUDED.scryfall_uri,
                text_hash = EXCLUDED.text_hash,
                color_identity = EXCLUDED.color_identity,
                price_usd = EXCLUDED.price_usd,
                price_eur = EXCLUDED.price_eur,
                cmc = EXCLUDED.cmc,
                game_changer = EXCLUDED.game_changer,
                legal_commander = EXCLUDED.legal_commander,
                updated_at = now()
        """, card_rows)

    work = new_cards + changed_cards
    if work:
        #collect every line from every new or changed card so the model runs
        #once over one big batch instead of once per card
        texts = []
        owners = []  #texts[i] belongs to work[owners[i]]
        for i, (c, h) in enumerate(work):
            for line in split_lines(c):
                texts.append(line)
                owners.append(i)

        #imported down here so the nothing-changed runs never pay the slow
        #torch import, it takes longer than the entire rest of the script
        print("loading the model (downloads ~1.2gb the very first time)...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
        print("embedding " + str(len(texts)) + " lines, this is the slow part...")
        embs = model.encode(texts, batch_size=64, show_progress_bar=True,
                            normalize_embeddings=True, prompt=EMBED_PROMPT)

        #all the writes ride in one transaction (one commit at the very end),
        #so a crash halfway through leaves the database exactly how it was.
        #executemany batches the rows into pipelined round trips, one insert
        #at a time would take half an hour from github's servers on the seed
        with conn.cursor() as cur:
            if model_changed:
                #the truncate lives down here on purpose: doing it before the
                #slow encode above would hold the table lock the whole time
                #and hang every search on the site. the alter resizes the
                #column, the old model was 384 dims
                cur.execute("TRUNCATE lines")
                cur.execute("ALTER TABLE lines ALTER COLUMN embedding TYPE vector(" + str(EMBED_DIMS) + ")")
            #changed cards get their old lines thrown out and rebuilt fresh
            elif changed_cards:
                old_ids = []
                for c, h in changed_cards:
                    old_ids.append((c["oracle_id"],))
                cur.executemany("DELETE FROM lines WHERE oracle_id = %s", old_ids)

            rows = []
            for j, text in enumerate(texts):
                c = work[owners[j]][0]
                rows.append((c["oracle_id"], text, embs[j]))
            print("writing " + str(len(rows)) + " lines...")
            cur.executemany("INSERT INTO lines (oracle_id, line_text, embedding) VALUES (%s, %s, %s)", rows)

    #deleting a card cascades to its lines, so this cleans up everything
    if stale:
        print("removing " + str(len(stale)) + " cards that are gone or filtered out now...")
        gone = []
        for oid in stale:
            gone.append((oid,))
        with conn.cursor() as cur:
            cur.executemany("DELETE FROM cards WHERE oracle_id = %s", gone)

    if work or stale:
        #recount how common every line is. its one group by over ~61k rows,
        #way easier than trying to patch the counts incrementally
        print("recounting how common every line is...")
        conn.execute("TRUNCATE line_stats")
        conn.execute("INSERT INTO line_stats SELECT line_text, count(*) FROM lines GROUP BY line_text")

    #remember which bulk file this was so tomorrow's run can skip it, and
    #which model made the vectors so the next swap rebuilds automatically
    conn.execute("""
        INSERT INTO meta (key, value) VALUES ('scryfall_updated_at', %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
    """, (updated_at,))
    conn.execute("""
        INSERT INTO meta (key, value) VALUES ('embed_model', %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
    """, (EMBED_MODEL,))
    conn.commit()
    conn.close()
    print("done! " + str(len(new_cards)) + " added, " + str(len(changed_cards)) + " updated, " + str(len(stale)) + " removed")


if __name__ == "__main__":
    main()
