#the actual website. the old version loaded a 90mb embedding matrix into
#memory at startup, this one just asks postgres. the embeddings live in the
#database and pgvector does the similarity math right where the data is, so
#this process stays tiny and never touches torch

import math

from flask import Flask, render_template, request

from db import pool

app = Flask(__name__)

#the display columns the frontend needs, so every query grabs the same set
CARD_FIELDS = "oracle_id, name, mana_cost, type_line, oracle_text, image, scryfall_uri"


#lines like "Flying" appear on thousands of cards, and if we don't do anything
#about it every flying creature "matches" every other flying creature at 100%
#and the results are useless. so common lines get weighted down when ranking.
#basically a homemade version of idf from search engines
def line_weight(count):
    return 1.0 / (1.0 + math.log(count))


def find_card(query):
    q = query.strip()
    with pool.connection() as conn:
        #exact match first, then startswith, then anywhere in the name
        row = conn.execute("SELECT " + CARD_FIELDS + " FROM cards WHERE lower(name) = lower(%s)", (q,)).fetchone()
        if row:
            return row
        row = conn.execute("SELECT " + CARD_FIELDS + " FROM cards WHERE name ILIKE %s ORDER BY name LIMIT 1", (q + "%",)).fetchone()
        if row:
            return row
        row = conn.execute("SELECT " + CARD_FIELDS + " FROM cards WHERE name ILIKE %s ORDER BY name LIMIT 1", ("%" + q + "%",)).fetchone()
        if row:
            return row
        #last resort, trigrams catch close spellings like "lightnig bolt".
        #the % operator means "similar enough to bother" (so garbage queries
        #still return nothing) and <-> sorts by closest. %% because psycopg
        #uses % for parameters
        return conn.execute("SELECT " + CARD_FIELDS + " FROM cards WHERE name %% %s ORDER BY name <-> %s LIMIT 1", (q, q)).fetchone()


def find_similar(oracle_id, offset=0, how_many=20):
    best = {}  #other card's oracle_id -> (weighted score, real similarity, our line, their line)
    with pool.connection() as conn:
        #the query card's lines are already embedded in the database, so the
        #model never runs at search time. grab them with their idf counts in one go
        qlines = conn.execute("""
            SELECT l.line_text, l.embedding, coalesce(s.count, 1) AS count
            FROM lines l LEFT JOIN line_stats s ON s.line_text = l.line_text
            WHERE l.oracle_id = %s
        """, (oracle_id,)).fetchall()

        for ql in qlines:
            w = line_weight(ql["count"])
            #<=> is pgvector's cosine distance, so similarity is 1 minus it.
            #grab way more than we need since a bunch will get merged per card
            matches = conn.execute("""
                SELECT oracle_id, line_text, 1 - (embedding <=> %s) AS sim
                FROM lines
                WHERE oracle_id <> %s
                ORDER BY embedding <=> %s
                LIMIT 400
            """, (ql["embedding"], oracle_id, ql["embedding"])).fetchall()
            for m in matches:
                score = m["sim"] * w
                if m["oracle_id"] not in best or score > best[m["oracle_id"]][0]:
                    best[m["oracle_id"]] = (score, m["sim"], ql["line_text"], m["line_text"])

        ranked = sorted(best.items(), key=lambda x: x[1][0], reverse=True)
        has_more = len(ranked) > offset + how_many
        page = ranked[offset:offset + how_many]

        #one query for the display info of just the cards on this page
        info = {}
        ids = [oid for oid, stuff in page]
        if ids:
            for row in conn.execute("SELECT " + CARD_FIELDS + " FROM cards WHERE oracle_id = ANY(%s)", (ids,)):
                info[row["oracle_id"]] = row

    results = []
    for oid, (score, sim, our_line, their_line) in page:
        c = info[oid]
        results.append({
            "name": c["name"],
            "mana_cost": c["mana_cost"],
            "type_line": c["type_line"],
            "image": c["image"],
            "scryfall_uri": c["scryfall_uri"],
            "percent": int(round(sim * 100)),
            "our_line": our_line,
            "their_line": their_line,
        })
    return results, has_more


@app.route("/")
def home():
    query = request.args.get("q", "")
    if not query:
        return render_template("index.html")

    card = find_card(query)
    if card is None:
        return render_template("index.html", query=query, not_found=True)

    results, has_more = find_similar(card["oracle_id"])
    #the template wants the full rules text under "text" like the old index had
    card["text"] = card["oracle_text"]
    return render_template("index.html", query=query, card=card, results=results, has_more=has_more)


#the load more button on the results page calls this and gets json back
@app.route("/more")
def more():
    query = request.args.get("q", "")
    offset = int(request.args.get("offset", 0))
    card = find_card(query)
    if card is None:
        return {"results": [], "has_more": False}
    results, has_more = find_similar(card["oracle_id"], offset)
    return {"results": results, "has_more": has_more}


#the search bar calls this while you type to fill the suggestion dropdown.
#names that start with what you typed come first, then names with it anywhere,
#then trigram matches to catch typos
@app.route("/suggest")
def suggest():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return {"names": []}
    names = []
    with pool.connection() as conn:
        for row in conn.execute("SELECT name FROM cards WHERE name ILIKE %s ORDER BY name LIMIT 8", (q + "%",)):
            names.append(row["name"])
        if len(names) < 8:
            for row in conn.execute("SELECT name FROM cards WHERE name ILIKE %s ORDER BY name LIMIT 8", ("%" + q + "%",)):
                if row["name"] not in names:
                    names.append(row["name"])
                    if len(names) == 8:
                        break
        #if substring matching didnt fill the list, fuzzy matching tops it up
        if len(names) < 8:
            for row in conn.execute("SELECT name FROM cards WHERE name %% %s ORDER BY name <-> %s LIMIT 8", (q, q)):
                if row["name"] not in names:
                    names.append(row["name"])
                    if len(names) == 8:
                        break
    return {"names": names}


if __name__ == "__main__":
    app.run(debug=True)
