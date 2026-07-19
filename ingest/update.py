#the daily updater. asks scryfall if their bulk card file changed, and if it
#did, only does the heavy embedding work for cards that are actually new or
#have different text than last time. most days thats a handful of cards, so
#the run finishes in seconds after the download.
#
#prices take a second, bigger download: the oracle file carries one price
#per card (whatever printing scryfall prefers), so the default_cards file
#(every printing) gets streamed through to find each card's cheapest paper
#printing instead.
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
import ijson
from pgvector.psycopg import register_vector

from common.cards import HEADERS, keep_card, split_lines, get_text, get_image, get_back_image
from common.concept import CALIBRATION as CONCEPT_CALIBRATION

BULK_URL = "https://api.scryfall.com/bulk-data"
DOWNLOAD_FILE = "oracle-cards.json"
PRICES_FILE = "default-cards.json"

#my fine tuned embeddinggemma (a sentence-transformers model). it sits in a
#private repo on hugging face, so HF_TOKEN has to be set or the download 401s.
#the prompt was glued to the front of every line during training, encoding
#without it gives useless vectors. swapping EMBED_MODEL for something else
#makes the next run rebuild every vector on its own
EMBED_MODEL = "BallchinianMan/mtg-tuned-embeddinggemma-300m"
EMBED_PROMPT = "task: sentence similarity | query: "
EMBED_DIMS = 768

#axis 1's calibration map: raw cosine -> the percent the site shows,
#piecewise linear through hand-judged pairs. raw cosine is arbitrary per
#model, so this map is ANCHORED TO THE MODEL ABOVE and lives right next to
#it - swapping models means re-judging these anchors along with rebuilding
#the vectors. the load-bearing anchor is 0.895 -> 80: the quality boundary
#the old raw-90 cutoff actually guarded (int(round()) let 89.5 through) now
#reads as 80 and exactly the same set of cards passes. identical text stays
#100 (nothing that isn't identical may show 100), the flagship match
#(rhystic/remora, raw .97) lands low 90s, and the "same shell, different
#payload" band (raw ~.85) drops visibly under the gate.
#
#both maps ride to the website through the meta table (written in main
#below, next to the model name they belong to), so the site and the
#pipeline can never disagree about what a percent means
MECH_CALIBRATION = [(0.0, 0), (0.50, 30), (0.70, 45), (0.85, 65), (0.895, 80), (0.97, 92), (1.0, 100)]


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


def download_bulk(item, path):
    #stream the file straight to disk instead of holding the raw json in
    #memory on top of the parsed version. item is one entry from the
    #bulk-data listing, it knows its own size
    print("downloading " + item["download_uri"])
    print("(its about " + str(item.get("size", 0) // (1024 * 1024)) + "mb so this can take a while)")
    for attempt in range(3):
        try:
            with requests.get(item["download_uri"], headers=HEADERS, timeout=300, stream=True) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            return
        except Exception as e:
            if attempt == 2:
                raise
            print("download failed (" + str(e) + "), retrying...")
            time.sleep(10)


def finish_price(prices, keys):
    #the cheapest finish (nonfoil, foil, etched) of one printing. scryfall
    #sends strings like "0.25" or None for finishes that dont exist
    best = None
    for k in keys:
        p = prices.get(k)
        if p is not None:
            p = float(p)
            if best is None or p < best:
                best = p
    return best


def cheapest_prices(item):
    #oracle_id -> [usd, eur], the lowest price across every paper printing,
    #plus oracle_id -> the earliest released_at, when the card first existed.
    #the default_cards file is a couple of gigabytes, so ijson streams it one
    #card at a time instead of parsing the whole thing into memory the way
    #the oracle file is
    download_bulk(item, PRICES_FILE)
    print("scanning every printing for the cheapest price and first release...")
    best = {}
    debut = {}
    printings = 0
    with open(PRICES_FILE, "rb") as f:
        for c in ijson.items(f, "item"):
            printings += 1
            oid = c.get("oracle_id")
            if not oid and c.get("card_faces"):
                oid = c["card_faces"][0].get("oracle_id")  #reversible cards keep it per face
            if not oid:
                continue
            #the debut counts every printing, even the ones the price hunt
            #skips below: a digital only debut is still the card's debut.
            #iso dates compare fine as strings
            rel = c.get("released_at")
            if rel and (oid not in debut or rel < debut[oid]):
                debut[oid] = rel
            #versions you cant actually buy as the real paper card dont
            #count: arena/mtgo printings, oversized promos, and memorabilia
            #(gold border world championship decks would underprice half the
            #expensive staples in the game)
            if c.get("digital") or c.get("oversized") or c.get("set_type") == "memorabilia":
                continue
            prices = c.get("prices", {})
            usd = finish_price(prices, ("usd", "usd_foil", "usd_etched"))
            eur = finish_price(prices, ("eur", "eur_foil", "eur_etched"))
            low = best.get(oid)
            if low is None:
                best[oid] = [usd, eur]
            else:
                if usd is not None and (low[0] is None or usd < low[0]):
                    low[0] = usd
                if eur is not None and (low[1] is None or eur < low[1]):
                    low[1] = eur
    os.remove(PRICES_FILE)
    print("checked " + str(printings) + " printings of " + str(len(best)) + " cards")
    return best, debut


def card_hash(card):
    #the name is part of the hash on purpose: clean_line swaps the card's own
    #name for "this card" inside the text, so a renamed card needs its lines
    #rebuilt even if the text itself looks the same
    return hashlib.sha256((card["name"] + "\n" + get_text(card)).encode("utf-8")).hexdigest()


def recompute_uniqueness(conn):
    #fills lines.nn_sim (how close the closest line on any OTHER card gets to
    #this one) and rolls the scores up into cards.uniqueness for the /unique
    #page. its the search query turned inside out: search asks "whats
    #closest", this asks "how far away is even the closest thing".
    #
    #everything gets recomputed from scratch whenever lines changed, same
    #philosophy as line_stats. incremental sounds tempting until you notice a
    #NEW card can make an OLD card less unique (it might be its new nearest
    #neighbor), and a DELETED card can make its old neighbors MORE unique, so
    #patching only the changed rows would quietly rot every score around them.
    #
    #and yes, this pulls every embedding out of postgres and does the math in
    #numpy, the exact thing the web app was rescued from. the difference is
    #WHERE: asking pgvector for 31k exact nearest neighbors takes ~370ms each
    #on the railway box (measured), call it three hours of pegged production
    #database, while one big matrix multiply on the ingest runner takes about
    #a minute. all-pairs work belongs next to the big cpu, one-query-at-a-time
    #work belongs next to the data. the web server still never touches this
    print("recomputing uniqueness scores...")
    import numpy as np  #late import like torch below, the no-op runs skip it

    ids = []
    owners = []      #row i belongs to card owners[i]
    vecs = []
    for lid, oid, vec in conn.execute("SELECT id, oracle_id, embedding FROM lines WHERE NOT whole"):
        ids.append(lid)
        owners.append(oid)
        #pgvector hands back its own Vector class, not a numpy array
        vecs.append(vec.to_numpy())
    if not ids:
        return  #empty lines table, nothing to score
    emb = np.asarray(vecs, dtype=np.float32)
    print("  pulled " + str(len(ids)) + " embeddings, multiplying...")

    #which rows belong to each card, so a card never counts as its own neighbor
    rows_of_card = {}
    for i, oid in enumerate(owners):
        rows_of_card.setdefault(oid, []).append(i)

    #the embeddings are normalized so cosine similarity is just a dot product.
    #block by block keeps the similarity matrix at ~100mb instead of 13gb
    nn_sim = np.zeros(len(ids), dtype=np.float32)
    block = 512
    for start in range(0, len(ids), block):
        sims = emb[start:start + block] @ emb.T
        for r in range(sims.shape[0]):
            sims[r, rows_of_card[owners[start + r]]] = -2.0  #below any real cosine
        nn_sim[start:start + block] = sims.max(axis=1)

    #COPY the scores into a temp table and update from there, one round trip
    #instead of 58k. the IS DISTINCT FROM means unchanged rows dont get
    #rewritten, which on a normal day is nearly all of them
    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE nn_tmp (id bigint PRIMARY KEY, nn_sim real) ON COMMIT DROP")
        with cur.copy("COPY nn_tmp (id, nn_sim) FROM STDIN") as copy:
            for i, lid in enumerate(ids):
                copy.write_row((lid, float(nn_sim[i])))
        cur.execute("""
            UPDATE lines l SET nn_sim = t.nn_sim
            FROM nn_tmp t
            WHERE l.id = t.id AND l.nn_sim IS DISTINCT FROM t.nn_sim
        """)

        #a card is as unique as its most isolated line. DISTINCT ON keeps one
        #row per card and the ORDER BY makes it the line with the lowest
        #nearest neighbor similarity, so a card with Flying plus one ability
        #nobody else has still counts as unique, the Flying line just never
        #wins the argmin
        cur.execute("""
            UPDATE cards c SET uniqueness = (1 - s.nn_sim)::real, unique_line = s.line_text
            FROM (SELECT DISTINCT ON (oracle_id) oracle_id, nn_sim, line_text
                  FROM lines
                  WHERE nn_sim IS NOT NULL
                  ORDER BY oracle_id, nn_sim ASC) s
            WHERE c.oracle_id = s.oracle_id
              AND (c.uniqueness IS DISTINCT FROM (1 - s.nn_sim)::real
                   OR c.unique_line IS DISTINCT FROM s.line_text)
        """)
    conn.commit()
    print("uniqueness done")


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

    #the calibration maps go into meta before the gate below, so even a
    #nothing-changed run leaves them in place for the website to read. the
    #site carries seed copies but the database's word wins, which is what
    #keeps a model swap atomic: new vectors and their new map arrive together
    for key, cal in (("mech_calibration", MECH_CALIBRATION),
                     ("concept_calibration", CONCEPT_CALIBRATION)):
        conn.execute("""
            INSERT INTO meta (key, value) VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """, (key, json.dumps(cal)))
    conn.commit()

    #vectors from two different models cant be compared with each other, so
    #if the database was embedded by anything other than the model above,
    #every line needs redoing this run
    row = conn.execute("SELECT value FROM meta WHERE key = 'embed_model'").fetchone()
    model_changed = row is None or row[0] != EMBED_MODEL
    if model_changed:
        print("embedding model changed, this run rebuilds every vector (the slow full reseed)")

    print("asking scryfall where the bulk files live...")
    bulk = None
    prices_bulk = None
    for item in get_with_retries(BULK_URL).json()["data"]:
        #oracle_cards = one entry per unique card instead of every single
        #printing. default_cards = every printing, the price scan needs those
        if item["type"] == "oracle_cards":
            bulk = item
        if item["type"] == "default_cards":
            prices_bulk = item
    updated_at = bulk["updated_at"]

    #the gate: if we already processed this exact bulk file, stop right here.
    #this is what makes rerunning the workflow basically free. unless the
    #model changed, then theres a full rebuild to do either way
    row = conn.execute("SELECT value FROM meta WHERE key = 'scryfall_updated_at'").fetchone()
    if not model_changed and row and row[0] == updated_at:
        #the bulk file might be old news while the uniqueness scores arent:
        #the first run after the /unique feature shipped, or a recompute that
        #died halfway. any line without a score means theres finishing to do
        if conn.execute("SELECT 1 FROM lines WHERE nn_sim IS NULL AND NOT whole LIMIT 1").fetchone():
            recompute_uniqueness(conn)
        else:
            print("already processed the bulk file from " + updated_at + ", nothing to do")
        conn.close()
        return

    download_bulk(bulk, DOWNLOAD_FILE)
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

    cheapest, debut = cheapest_prices(prices_bulk)

    #what do we already have? oracle_id -> hash of the text we embedded last time
    have = {}
    for oracle_id, text_hash in conn.execute("SELECT oracle_id, text_hash FROM cards"):
        have[str(oracle_id)] = text_hash

    if model_changed:
        #forget the stored hashes so every card counts as new and gets
        #embedded again. the old vectors stay put for now, the site keeps
        #searching on them while the new ones compute
        have = {}

    new_cards = []
    changed_cards = []
    unchanged = 0
    card_rows = []  #every kept card, for the upsert below
    for c in cards:
        h = card_hash(c)
        #the cheapest printing's price when the scan found one, falling back
        #to the oracle file's own (scryfall's preferred printing). strings,
        #floats or None, postgres takes any of them into a numeric column
        prices = c.get("prices", {})
        low = cheapest.get(c["oracle_id"], [None, None])
        usd = low[0] if low[0] is not None else prices.get("usd")
        eur = low[1] if low[1] is not None else prices.get("eur")
        #the earliest printing's date from the scan, falling back to the
        #oracle file's own (scryfall's preferred printing, could be a reprint)
        rel = debut.get(c["oracle_id"]) or c.get("released_at")
        card_rows.append((c["oracle_id"], c["name"], c.get("mana_cost", ""), c.get("type_line", ""),
                          get_text(c), get_image(c), c.get("scryfall_uri", ""), h,
                          "".join(c.get("color_identity", [])), usd, eur,
                          c.get("cmc", 0), c.get("game_changer", False),
                          c.get("legalities", {}).get("commander") == "legal",
                          c.get("layout", "normal"), get_back_image(c), c.get("edhrec_rank"), rel))
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

    #every card row gets offered every run, not just the new and changed ones.
    #prices move daily and wizards edits the game changer list now and then,
    #so waiting for a rules text change would leave those stale forever. the
    #WHERE on the conflict clause skips rows where nothing actually differs,
    #otherwise every run rewrites all ~31k rows (a day of dead tuples and wal
    #for the autovacuum to mop up) just to store the same values. it also
    #makes updated_at mean "last actually changed". the slow embedding work
    #below still only happens when text changed
    print("writing " + str(len(card_rows)) + " card rows (keeps prices and filters fresh)...")
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO cards (oracle_id, name, mana_cost, type_line, oracle_text, image, scryfall_uri, text_hash,
                               color_identity, price_usd, price_eur, cmc, game_changer, legal_commander,
                               layout, image_back, edhrec_rank, released_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
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
                layout = EXCLUDED.layout,
                image_back = EXCLUDED.image_back,
                edhrec_rank = EXCLUDED.edhrec_rank,
                released_at = EXCLUDED.released_at,
                updated_at = now()
            WHERE (cards.name, cards.mana_cost, cards.type_line, cards.oracle_text, cards.image,
                   cards.scryfall_uri, cards.text_hash, cards.color_identity, cards.price_usd,
                   cards.price_eur, cards.cmc, cards.game_changer, cards.legal_commander,
                   cards.layout, cards.image_back, cards.edhrec_rank, cards.released_at)
                  IS DISTINCT FROM
                  (EXCLUDED.name, EXCLUDED.mana_cost, EXCLUDED.type_line, EXCLUDED.oracle_text, EXCLUDED.image,
                   EXCLUDED.scryfall_uri, EXCLUDED.text_hash, EXCLUDED.color_identity, EXCLUDED.price_usd,
                   EXCLUDED.price_eur, EXCLUDED.cmc, EXCLUDED.game_changer, EXCLUDED.legal_commander,
                   EXCLUDED.layout, EXCLUDED.image_back, EXCLUDED.edhrec_rank, EXCLUDED.released_at)
        """, card_rows)

    work = new_cards + changed_cards
    if work:
        #collect every line from every new or changed card so the model runs
        #once over one big batch instead of once per card
        texts = []
        faces = []
        wholes = []
        owners = []  #texts[i] belongs to work[owners[i]]
        for i, (c, h) in enumerate(work):
            card_lines = split_lines(c)
            for line, face in card_lines:
                texts.append(line)
                faces.append(face)
                wholes.append(False)
                owners.append(i)
            #multi-line cards also get one whole-card row (all their cleaned
            #lines together), retrieval material for the line-merging blind
            #spot. single-line cards would just duplicate their line
            if len(card_lines) > 1:
                texts.append("\n".join(line for line, face in card_lines))
                faces.append(0)
                wholes.append(True)
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
                rows.append((c["oracle_id"], text, embs[j], faces[j], wholes[j]))
            print("writing " + str(len(rows)) + " lines...")
            cur.executemany("INSERT INTO lines (oracle_id, line_text, embedding, face, whole) VALUES (%s, %s, %s, %s, %s)", rows)

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
        #way easier than trying to patch the counts incrementally.
        #
        #counted per SHAPE, not per exact text: a run of mana symbols collapses
        #to one placeholder first, so "Overload {4}{R}" and "Overload {2}{R}"
        #share a bucket. counting exact text let any keyword with a varying cost
        #dodge the idf weighting, fragmenting into one and two card texts that
        #each drew the full 1.0 weight a unique ability gets (overload: 27
        #card-lines, 22 texts, biggest on 2), which is how Vandalblast matched
        #Dynacharge at 99% on the keyword alone. still KEYED by exact text,
        #which is what the search joins on. no braces, no change: Flying = 2517
        print("recounting how common every line is...")
        conn.execute("TRUNCATE line_stats")
        conn.execute(r"""
            INSERT INTO line_stats
            SELECT line_text, sum(n) OVER (PARTITION BY shape)
            FROM (
                SELECT line_text, count(*) AS n,
                       regexp_replace(line_text, '(\{[^}]*\})+', '{C}', 'g') AS shape
                FROM lines WHERE NOT whole GROUP BY line_text
            ) t
        """)

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

    #uniqueness runs after the commit above on purpose: its derived data, so
    #if it dies halfway the database is still fully consistent and the NULL
    #check at the gate finishes the job on the next run. the NULL check here
    #catches databases from before the /unique feature even on days when no
    #cards changed
    if work or stale or conn.execute("SELECT 1 FROM lines WHERE nn_sim IS NULL AND NOT whole LIMIT 1").fetchone():
        recompute_uniqueness(conn)

    conn.close()
    print("done! " + str(len(new_cards)) + " added, " + str(len(changed_cards)) + " updated, " + str(len(stale)) + " removed")


if __name__ == "__main__":
    main()
