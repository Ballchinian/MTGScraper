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

    print("asking scryfall where the bulk file lives...")
    bulk = None
    for item in get_with_retries(BULK_URL).json()["data"]:
        #oracle_cards = one entry per unique card instead of every single printing
        if item["type"] == "oracle_cards":
            bulk = item
    updated_at = bulk["updated_at"]

    #the gate: if we already processed this exact bulk file, stop right here.
    #this is what makes rerunning the workflow basically free
    row = conn.execute("SELECT value FROM meta WHERE key = 'scryfall_updated_at'").fetchone()
    if row and row[0] == updated_at:
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

    new_cards = []
    changed_cards = []
    unchanged = 0
    for c in cards:
        h = card_hash(c)
        old = have.get(c["oracle_id"])
        if old is None:
            new_cards.append((c, h))
        elif old != h:
            changed_cards.append((c, h))
        else:
            unchanged += 1
    print(str(len(new_cards)) + " new, " + str(len(changed_cards)) + " changed, " + str(unchanged) + " unchanged")

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
        print("loading the model (downloads ~90mb the very first time)...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("embedding " + str(len(texts)) + " lines, this is the slow part...")
        embs = model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)

        #all the writes ride in one transaction, so a crash halfway through
        #leaves the database exactly how it was
        with conn.cursor() as cur:
            for c, h in work:
                cur.execute("""
                    INSERT INTO cards (oracle_id, name, mana_cost, type_line, oracle_text, image, scryfall_uri, text_hash, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (oracle_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        mana_cost = EXCLUDED.mana_cost,
                        type_line = EXCLUDED.type_line,
                        oracle_text = EXCLUDED.oracle_text,
                        image = EXCLUDED.image,
                        scryfall_uri = EXCLUDED.scryfall_uri,
                        text_hash = EXCLUDED.text_hash,
                        updated_at = now()
                """, (c["oracle_id"], c["name"], c.get("mana_cost", ""), c.get("type_line", ""),
                      get_text(c), get_image(c), c.get("scryfall_uri", ""), h))

            #changed cards get their old lines thrown out and rebuilt fresh
            for c, h in changed_cards:
                cur.execute("DELETE FROM lines WHERE oracle_id = %s", (c["oracle_id"],))

            rows = []
            for j, text in enumerate(texts):
                c = work[owners[j]][0]
                rows.append((c["oracle_id"], text, embs[j]))
            cur.executemany("INSERT INTO lines (oracle_id, line_text, embedding) VALUES (%s, %s, %s)", rows)

            #recount how common every line is. its one group by over ~61k rows,
            #way easier than trying to patch the counts incrementally
            cur.execute("TRUNCATE line_stats")
            cur.execute("INSERT INTO line_stats SELECT line_text, count(*) FROM lines GROUP BY line_text")

    #remember which bulk file this was so tomorrow's run can skip it
    conn.execute("""
        INSERT INTO meta (key, value) VALUES ('scryfall_updated_at', %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
    """, (updated_at,))
    conn.commit()
    conn.close()
    print("done! " + str(len(new_cards)) + " added, " + str(len(changed_cards)) + " updated")


if __name__ == "__main__":
    main()
