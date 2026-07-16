#pulls scryfall tagger's community tags (the official oracle_tags bulk file,
#one download, updated daily) into card_tags + tags for the concept axis.
#run it from the repo root like the updater:
#    python -m ingest.tags
#with DATABASE_URL set. reruns are free: the meta gate skips the work unless
#scryfall published a newer bulk file.
#
#everything gets ingested except the trivia subtrees below - curation is a
#blocklist, not an allowlist, because rare tags are the high-precision signal
#(two cards sharing "wheel" says more than two sharing "removal") and idf
#weighting already mutes the mega-broad ones

import os
import sys
import json

import psycopg

from ingest.update import BULK_URL, get_with_retries

#subtree roots whose tags say nothing about what a card DOES: naming schemes,
#set-design cycles (unrelated to the cycling mechanic, which has no bare tag
#of its own - taggers only tag interactions like synergy-cycling), wordplay,
#vanilla-ness, templating syntax, and type-line trivia
BLOCKED_ROOTS = [
    "card-names",
    "cycle",
    "alliteration",
    "flavors-of-vanilla",
    "type-errata",
    "unique-type-line",
    "namesake-spell",
    "intervening-if-clause",
]


def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("set DATABASE_URL first (the postgres connection string)")
        sys.exit(1)

    conn = psycopg.connect(db_url)
    schema_path = os.path.join(os.path.dirname(__file__), "..", "common", "schema.sql")
    with open(schema_path, encoding="utf-8") as f:
        conn.execute(f.read())
    conn.commit()

    print("asking scryfall where the tag bulk file lives...")
    bulk = None
    for item in get_with_retries(BULK_URL).json()["data"]:
        if item["type"] == "oracle_tags":
            bulk = item
    updated_at = bulk["updated_at"]

    #same gate as the card updater: seen this exact file already, stop. an
    #empty card_tags table means a first run (or a died-halfway one), do the
    #work anyway
    row = conn.execute("SELECT value FROM meta WHERE key = 'tagger_updated_at'").fetchone()
    if (row and row[0] == updated_at
            and conn.execute("SELECT 1 FROM card_tags LIMIT 1").fetchone()
            and conn.execute("SELECT 1 FROM card_tag_norms LIMIT 1").fetchone()
            and conn.execute("SELECT 1 FROM cards WHERE concept_uniqueness IS NOT NULL LIMIT 1").fetchone()):
        print("already processed the tag file from " + updated_at + ", nothing to do")
        conn.close()
        return

    print("downloading " + bulk["download_uri"] + " (~18mb)")
    all_tags = get_with_retries(bulk["download_uri"]).json()
    print("scryfall gave us " + str(len(all_tags)) + " oracle tags")

    #walk the hierarchy down from every blocked root and drop whole subtrees
    by_id = {t["id"]: t for t in all_tags}
    blocked_ids = set()
    frontier = [t["id"] for t in all_tags if t["slug"] in BLOCKED_ROOTS]
    while frontier:
        tid = frontier.pop()
        if tid in blocked_ids:
            continue
        blocked_ids.add(tid)
        frontier.extend(by_id[tid].get("child_ids", []))
    kept = [t for t in all_tags if t["id"] not in blocked_ids]
    print("blocked " + str(len(blocked_ids)) + " trivia tags, kept " + str(len(kept)))

    #tagger knows cards this database filters out (un-sets, digital only), so
    #only links to cards we actually have make it in
    ours = set()
    for (oid,) in conn.execute("SELECT oracle_id FROM cards"):
        ours.add(str(oid))

    links = set()
    for t in kept:
        for tagging in t.get("taggings", []):
            if tagging["oracle_id"] in ours:
                links.add((tagging["oracle_id"], t["slug"]))

    count_of = {}
    for oid, slug in links:
        count_of[slug] = count_of.get(slug, 0) + 1

    #parents only count if they survived the blocklist themselves
    kept_ids = {t["id"] for t in kept}
    tag_rows = []
    for t in kept:
        parents = [by_id[p]["slug"] for p in t.get("parent_ids", []) if p in kept_ids]
        tag_rows.append((t["slug"], parents, count_of.get(t["slug"], 0), t.get("description") or ""))

    #full rebuild in one transaction, so a crash leaves the old data standing
    with conn.cursor() as cur:
        cur.execute("TRUNCATE card_tags, tags, card_tag_norms")
        with cur.copy("COPY card_tags (oracle_id, tag) FROM STDIN") as copy:
            for oid, slug in links:
                copy.write_row((oid, slug))
        with cur.copy("COPY tags (tag, parents, card_count, description) FROM STDIN") as copy:
            for r in tag_rows:
                copy.write_row(r)
        #the derived weights the concept scorer reads: idf per tag, then each
        #card's vector length from those. baked here so every query agrees on
        #them and none has to recompute 31k norms
        cur.execute("UPDATE tags SET idf = ln((SELECT count(*) FROM cards)::float / greatest(card_count, 1))")
        cur.execute("""
            INSERT INTO card_tag_norms
            SELECT ct.oracle_id, sqrt(sum(t.idf * t.idf))
            FROM card_tags ct JOIN tags t ON t.tag = ct.tag
            GROUP BY ct.oracle_id
        """)
        cur.execute("""
            INSERT INTO meta (key, value) VALUES ('tagger_updated_at', %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """, (updated_at,))
    conn.commit()

    #concept uniqueness, the tag-space counterpart of lines.nn_sim: 1 minus
    #the best cosine any OTHER card's idf-weighted tag vector manages. same
    #all-pairs-in-blocks trick as the ingest's uniqueness pass. untagged
    #cards stay NULL, unknown is not the same as unique
    print("computing concept uniqueness...")
    import math
    import numpy as np

    n_cards = len(ours)
    idf = {slug: math.log(n_cards / max(count_of.get(slug, 1), 1)) for slug, _, _, _ in tag_rows}
    tag_col = {slug: i for i, slug in enumerate(sorted(idf))}
    card_row = {}
    for oid, slug in links:
        card_row.setdefault(oid, len(card_row))
    m = np.zeros((len(card_row), len(tag_col)), dtype=np.float32)
    for oid, slug in links:
        m[card_row[oid], tag_col[slug]] = idf[slug]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1  #cant happen unless a tag covers every card, but nan poisons everything
    m /= norms

    best = np.zeros(len(card_row), dtype=np.float32)
    block = 256
    for start in range(0, len(card_row), block):
        sims = m[start:start + block] @ m.T
        for r in range(sims.shape[0]):
            sims[r, start + r] = -2.0  #a card is not its own neighbor
        best[start:start + block] = sims.max(axis=1)

    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE cu_tmp (oracle_id uuid PRIMARY KEY, cu real) ON COMMIT DROP")
        with cur.copy("COPY cu_tmp (oracle_id, cu) FROM STDIN") as copy:
            for oid, i in card_row.items():
                copy.write_row((oid, float(1.0 - best[i])))
        cur.execute("UPDATE cards c SET concept_uniqueness = t.cu FROM cu_tmp t WHERE c.oracle_id = t.oracle_id")
        cur.execute("UPDATE cards SET concept_uniqueness = NULL WHERE oracle_id NOT IN (SELECT oracle_id FROM card_tags)")
    conn.commit()

    covered = conn.execute("SELECT count(DISTINCT oracle_id) FROM card_tags").fetchone()[0]
    total = conn.execute("SELECT count(*) FROM cards").fetchone()[0]
    conn.close()
    print("done! " + str(len(links)) + " card-tag links across " + str(len(tag_rows)) + " tags, "
          + str(covered) + "/" + str(total) + " cards have at least one tag")


if __name__ == "__main__":
    main()
