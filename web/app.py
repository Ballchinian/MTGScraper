#the actual website. the old version loaded a 90mb embedding matrix into
#memory at startup, this one just asks postgres. the embeddings live in the
#database and pgvector does the similarity math right where the data is, so
#this process stays tiny and never touches torch

import re
import math

from flask import Flask, render_template, request

from db import pool

app = Flask(__name__)

#the display columns the frontend needs, so every query grabs the same set
CARD_FIELDS = "oracle_id, name, mana_cost, type_line, oracle_text, image, scryfall_uri, price_usd"

#the choices in the type filter dropdown. also acts as a whitelist so
#nothing weird from the url ends up inside a LIKE pattern
CARD_TYPES = ["Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Planeswalker", "Battle", "Land"]


#copied from common/cards.py because railway only deploys the web folder.
#it has to stay identical to what the ingest used, otherwise the line picker
#cant match the lines shown on the page back to their rows in the database
def clean_line(line, card_name):
    line = re.sub(r"\(.*?\)", "", line)
    line = line.replace(card_name, "this card")
    if "," in card_name:
        line = line.replace(card_name.split(",")[0], "this card")
    return line.strip()


#lines like "Flying" appear on thousands of cards, and if we don't do anything
#about it every flying creature "matches" every other flying creature at 100%
#and the results are useless. so common lines get weighted down when ranking.
#basically a homemade version of idf from search engines.
#
#the old curve started punishing at count 2, which buried exactly the best
#results: a line shared by 2 cards means someone printed a functional reprint
#of it, and that reprint is the match people came for. so nothing gets
#punished until a line is on more than 5 cards, then it falls off gently
def line_weight(count):
    if count <= 5:
        return 1.0
    return 1.0 / (1.0 + math.log10(count / 5.0))


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


#all the url params past q and offset live in these little readers. home()
#and /more both use them, so the load more button sees exactly the same
#filters and picked lines as the page it's glued onto

def read_filters():
    f = {}
    #the color checkboxes arrive as repeated params like colors=W&colors=U.
    #only real color letters get through, which matters because they end up
    #inside a regex below
    letters = ""
    for c in request.args.getlist("colors"):
        if len(c) == 1 and c in "WUBRG" and c not in letters:
            letters += c
    f["colors"] = letters
    f["pmin"] = read_number(request.args.get("pmin", ""))
    f["pmax"] = read_number(request.args.get("pmax", ""))
    f["mvmin"] = read_number(request.args.get("mvmin", ""))
    f["mvmax"] = read_number(request.args.get("mvmax", ""))
    t = request.args.get("type", "")
    if t not in CARD_TYPES:
        t = ""
    f["type"] = t
    f["gc"] = request.args.get("gc") == "1"
    f["legal"] = request.args.get("legal") == "1"
    return f


def read_number(s):
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def read_min():
    #minimum match percent. 90 by default: on the fine tuned model the real
    #matches sit in the 90s, while "same shell, different payload" matches
    #(landfall gain life vs landfall proliferate) hover in the low-to-mid
    #80s. under a price sort the match percent is the only quality gate
    #left, so cheap half matches sorting to the top makes the site look
    #broken. everything below the line still exists, it just pages in after
    #the "show weaker matches" button instead of being thrown away
    try:
        m = int(request.args.get("min", 90))
    except ValueError:
        m = 90
    return max(0, min(m, 100))


def read_sort():
    s = request.args.get("sort", "")
    if s not in ("cheap", "pricey"):
        s = "best"
    return s


def read_picked():
    #the lines param holds indexes into the searched card's rules text, like
    #"0,2" for the first and third line, set by clicking lines on the page
    picked = set()
    for part in request.args.get("lines", "").split(","):
        if part.strip().isdigit():
            picked.add(int(part.strip()))
    return picked


def build_lines(card, picked_idx):
    #the searched card's rules text as individual lines for the line picker.
    #searchable means the cleaned line is real text that lives in the lines
    #table (reminder-only lines clean down to nothing and cant be picked).
    #returns the display list and the cleaned texts of the picked lines
    shown = []
    picked = []
    for idx, raw in enumerate(card["oracle_text"].split("\n")):
        cleaned = clean_line(raw, card["name"])
        searchable = len(cleaned) >= 3
        selected = searchable and idx in picked_idx
        if selected:
            picked.append(cleaned)
        shown.append({"idx": idx, "text": raw, "searchable": searchable, "selected": selected})
    return shown, picked


def filter_sql(filters):
    #turns the filters into conditions on the cards table. returns a snippet
    #starting with AND so it glues straight onto the candidate query, plus
    #its parameters. filtering inside that query matters: the LIMIT applies
    #after the filter, so narrow searches dig deeper into the rankings
    #instead of cutting down an already cut off list
    where = ""
    params = []
    if filters["colors"]:
        #fits-within, like deckbuilding: every letter of the card's identity
        #must be one of the picked colors, and colorless always fits. safe to
        #build into a regex because read_filters only lets WUBRG through
        where += " AND c.color_identity ~ %s"
        params.append("^[" + filters["colors"] + "]*$")
    if filters["pmin"] is not None:
        where += " AND c.price_usd >= %s"
        params.append(filters["pmin"])
    if filters["pmax"] is not None:
        #cards with no known price fail both comparisons, so any price filter
        #quietly drops them, which is what a budget search wants
        where += " AND c.price_usd <= %s"
        params.append(filters["pmax"])
    if filters["mvmin"] is not None:
        where += " AND c.cmc >= %s"
        params.append(filters["mvmin"])
    if filters["mvmax"] is not None:
        where += " AND c.cmc <= %s"
        params.append(filters["mvmax"])
    if filters["type"]:
        where += " AND c.type_line ILIKE %s"
        params.append("%" + filters["type"] + "%")
    if filters["gc"]:
        where += " AND NOT c.game_changer"
    if filters["legal"]:
        where += " AND c.legal_commander"
    return where, params


def find_similar(oracle_id, picked, filters, min_pct, sort, offset=0, how_many=20, weak=False):
    #every candidate card keeps all its matching line pairs now instead of
    #just the best one, so results can show "+2 more matching lines".
    #
    #cards split into two tiers around min_pct: the strong ones are the real
    #results, the weak ones sit behind the "show weaker matches" button.
    #weak=True means the caller is paging through that second tier
    pairs_by_card = {}  #other card's oracle_id -> list of (weighted score, real similarity, our line, their line)
    prices = {}         #other card's oracle_id -> usd price, for the price sorts
    where, fparams = filter_sql(filters)
    with pool.connection() as conn:
        #the query card's lines are already embedded in the database, so the
        #model never runs at search time. grab them with their idf counts in one go
        qlines = conn.execute("""
            SELECT l.line_text, l.embedding, coalesce(s.count, 1) AS count
            FROM lines l LEFT JOIN line_stats s ON s.line_text = l.line_text
            WHERE l.oracle_id = %s
        """, (oracle_id,)).fetchall()

        #if the user picked lines on the page, only search with those
        if picked:
            chosen = []
            for ql in qlines:
                if ql["line_text"] in picked:
                    chosen.append(ql)
            if chosen:
                qlines = chosen

        for ql in qlines:
            w = line_weight(ql["count"])
            #<=> is pgvector's cosine distance, so similarity is 1 minus it.
            #grab way more than we need since a bunch will get merged per card
            matches = conn.execute("""
                SELECT l.oracle_id, l.line_text, 1 - (l.embedding <=> %s) AS sim, c.price_usd
                FROM lines l JOIN cards c ON c.oracle_id = l.oracle_id
                WHERE l.oracle_id <> %s""" + where + """
                ORDER BY l.embedding <=> %s
                LIMIT 400
            """, [ql["embedding"], oracle_id] + fparams + [ql["embedding"]]).fetchall()
            for m in matches:
                if m["oracle_id"] not in pairs_by_card:
                    pairs_by_card[m["oracle_id"]] = []
                pairs_by_card[m["oracle_id"]].append((m["sim"] * w, m["sim"], ql["line_text"], m["line_text"]))
                prices[m["oracle_id"]] = m["price_usd"]

        #sort each card's pairs so pairs[0] is its best one, then rank the
        #cards by that best pair
        ranked = []
        for oid, pairs in pairs_by_card.items():
            pairs.sort(reverse=True)
            ranked.append((oid, pairs))
        ranked.sort(key=lambda x: x[1][0][0], reverse=True)

        #split at the minimum match line. the percent a card shows is its best
        #pair's real similarity, so that's what decides which side it lands on
        strong = []
        weak_tier = []
        for entry in ranked:
            if int(round(entry[1][0][1] * 100)) >= min_pct:
                strong.append(entry)
            else:
                weak_tier.append(entry)
        weak_count = len(weak_tier)
        if weak:
            wanted = weak_tier
        else:
            wanted = strong

        #price sorting happens after the filters, the ranking and the tier
        #split, so the percent keeps meaning what it always meant and weak
        #cards can never leapfrog strong ones. cards with no price sink to
        #the bottom and ties fall back to match score
        if sort != "best":
            priced = []
            unpriced = []
            for entry in wanted:
                if prices[entry[0]] is None:
                    unpriced.append(entry)
                else:
                    priced.append(entry)
            if sort == "cheap":
                priced.sort(key=lambda x: (float(prices[x[0]]), -x[1][0][0]))
            else:
                priced.sort(key=lambda x: (-float(prices[x[0]]), -x[1][0][0]))
            wanted = priced + unpriced

        has_more = len(wanted) > offset + how_many
        page = wanted[offset:offset + how_many]

        #one query for the display info of just the cards on this page
        info = {}
        ids = [oid for oid, pairs in page]
        if ids:
            for row in conn.execute("SELECT " + CARD_FIELDS + " FROM cards WHERE oracle_id = ANY(%s)", (ids,)):
                info[row["oracle_id"]] = row

    results = []
    for oid, pairs in page:
        c = info[oid]
        score, sim, our_line, their_line = pairs[0]
        #the extra pairs behind the "+n more matching lines" hint. pairs that
        #reuse a line already shown get skipped, so the count means genuinely
        #different abilities matched, not the same ability matching twice
        used_ours = [our_line]
        used_theirs = [their_line]
        more = []
        for p in pairs[1:]:
            if p[2] in used_ours or p[3] in used_theirs:
                continue
            used_ours.append(p[2])
            used_theirs.append(p[3])
            more.append('"' + p[3] + '" (' + str(int(round(p[1] * 100))) + '%) matches your "' + p[2] + '"')
        price = ""
        if c["price_usd"] is not None:
            price = "$" + str(c["price_usd"])
        results.append({
            "name": c["name"],
            "mana_cost": c["mana_cost"],
            "type_line": c["type_line"],
            "image": c["image"],
            "scryfall_uri": c["scryfall_uri"],
            "percent": int(round(sim * 100)),
            "our_line": our_line,
            "their_line": their_line,
            "price": price,
            "more_count": len(more),
            "more_text": "\n".join(more),
        })
    return results, has_more, weak_count


@app.route("/")
def home():
    query = request.args.get("q", "")
    if not query:
        return render_template("index.html")

    card = find_card(query)
    if card is None:
        return render_template("index.html", query=query, not_found=True)

    filters = read_filters()
    min_pct = read_min()
    sort = read_sort()
    card_lines, picked = build_lines(card, read_picked())

    results, has_more, weak_count = find_similar(card["oracle_id"], picked, filters, min_pct, sort)
    return render_template("index.html", query=query, card=card, card_lines=card_lines,
                           picked_count=len(picked), results=results, has_more=has_more,
                           weak_count=weak_count, min_pct=min_pct, types=CARD_TYPES)


#the load more button on the results page calls this and gets json back. it
#receives the page's whole query string, so filters and picked lines apply
@app.route("/more")
def more():
    query = request.args.get("q", "")
    offset = int(request.args.get("offset", 0))
    card = find_card(query)
    if card is None:
        return {"results": [], "has_more": False, "weak_count": 0}
    card_lines, picked = build_lines(card, read_picked())
    weak = request.args.get("weak") == "1"
    results, has_more, weak_count = find_similar(card["oracle_id"], picked, read_filters(), read_min(), read_sort(), offset, weak=weak)
    return {"results": results, "has_more": has_more, "weak_count": weak_count}


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
