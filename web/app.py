#the actual website. the old version loaded a 90mb embedding matrix into
#memory at startup, this one just asks postgres. the embeddings live in the
#database and pgvector does the similarity math right where the data is, so
#this process stays tiny and never touches torch

import re
import os
import math
import uuid
import json

from flask import Flask, render_template, request, redirect, abort

from db import pool
from prefix_words import PREFIX_WORDS

app = Flask(__name__)

#user reports from the results page (see the /feedback route). the table
#really lives in common/schema.sql, but that file ships with the ingest and
#railway only deploys the web folder, so the web app makes sure its own
#table exists, same reasoning as clean_line being copied below. names are
#snapshotted next to the ids because cards can vanish from the cards table
#before a report gets reviewed, and no foreign keys for the same reason
with pool.connection() as _conn:
    _conn.execute("""
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
        )
    """)

#the review page at /admin only exists when this is set in the environment
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")

#the display columns the frontend needs, so every query grabs the same set
CARD_FIELDS = "oracle_id, name, mana_cost, type_line, oracle_text, image, scryfall_uri, price_usd, layout, image_back"

#the choices in the type filter dropdown. also acts as a whitelist so
#nothing weird from the url ends up inside a LIKE pattern
CARD_TYPES = ["Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Planeswalker", "Battle", "Land"]

#the /unique page deals this many cards per request. one at a time, its the
#counterpart to scryfall's random card button
UNIQUE_PAGE = 1

#default minimum uniqueness (0-100, higher = more unique). a card's
#uniqueness is 100 minus the best match its most isolated line has anywhere
#else in the game, precomputed by the ingest into cards.uniqueness.
#
#measured on the real data (2026-07-13): the median card only scores ~8,
#because most cards have a near twin somewhere. 30 keeps the top ~2.4%
#(739 cards), which is where the genuinely deck-defining stuff lives, and
#nothing in the whole game clears 51, so the input's ceiling of 100 is
#theoretical. lowering the bar is one keystroke on the page
UNIQUE_DEFAULT = 30


#copied from common/cards.py because railway only deploys the web folder.
#it has to stay identical to what the ingest used, otherwise the line picker
#cant match the lines shown on the page back to their rows in the database
def clean_line(line, card_name):
    line = re.sub(r"\(.*?\)", "", line)
    #flavour prefixes go, exactly like the ingest side: die-roll rows, saga
    #chapters, and scryfall's catalog of ability/flavor words before a dash
    line = re.sub(r"^\d+(?:—\d+)?\s*\|\s*", "", line)
    line = re.sub(r"^[IVX]+(?:, [IVX]+)*\s+—\s+", "", line)
    m = re.match(r"^([^—•|]{1,40}?)\s+—\s+(?=\S)", line)
    if m and m.group(1) in PREFIX_WORDS:
        line = line[m.end():]
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


#copied from common/concept.py because railway only deploys the web folder,
#same story as clean_line. axis 2: how conceptually close two cards are,
#scored from the community tags the ingest bakes into card_tags/tags. the
#raw cosine lives in a compressed band, this map turns it into the percent
#the site shows, and the gate is written in displayed units on purpose
CALIBRATION = [(0.0, 0), (0.10, 35), (0.22, 55), (0.35, 70), (0.55, 82), (0.68, 90), (1.0, 100)]
MIN_CONCEPT = 80


def concept_display(raw):
    raw = max(0.0, min(1.0, raw))
    for (x0, y0), (x1, y1) in zip(CALIBRATION, CALIBRATION[1:]):
        if raw <= x1:
            return round(y0 + (y1 - y0) * (raw - x0) / (x1 - x0))
    return 100


def concept_raw_gate(pct):
    #the map walked backwards, so the displayed gate becomes a raw sql cutoff
    pct = max(0, min(100, pct))
    for (x0, y0), (x1, y1) in zip(CALIBRATION, CALIBRATION[1:]):
        if pct <= y1:
            return x0 + (x1 - x0) * (pct - y0) / (y1 - y0)
    return 1.0


#the mechanical axis wears a calibration map too, same shape as the concept
#one. raw cosine is arbitrary per model (this map is anchored to the tuned
#embeddinggemma the ingest embeds with - swapping models means re-anchoring),
#so the displayed percent is pinned to judged pairs instead. the load-bearing
#anchor is 0.895 -> 80: the quality boundary the old raw-90 cutoff actually
#guarded (int(round()) let 89.5 through) now READS as 80 and exactly the
#same set of cards passes. identical text stays 100 (nothing that isn't
#identical may show 100), the flagship match (rhystic/remora, raw .97) lands
#low 90s, and the "same shell, different payload" band (raw ~.85) drops
#visibly under the gate
MECH_CALIBRATION = [(0.0, 0), (0.50, 30), (0.70, 45), (0.85, 65), (0.895, 80), (0.97, 92), (1.0, 100)]


def mech_display(raw):
    raw = max(0.0, min(1.0, raw))
    for (x0, y0), (x1, y1) in zip(MECH_CALIBRATION, MECH_CALIBRATION[1:]):
        if raw <= x1:
            return round(y0 + (y1 - y0) * (raw - x0) / (x1 - x0))
    return 100


#the mechanics <-> concepts slider maps its detents to these axis-2 weights
#(a 5% step existed once and changed nothing visible, so it went). results
#are ordered by (1-w) * mech percent + w * concept percent, and once the
#slider moves the badge shows that same blend, so the list always reads in
#descending order of the number on it. 0 is the default and leaves the
#search exactly as it always was
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def concept_between(conn, oracle_a, oracle_b):
    #the calibrated concept percent between two specific cards, for the
    #feedback path. 0 when either card carries no tags at all
    norms = conn.execute("SELECT norm FROM card_tag_norms WHERE oracle_id IN (%s, %s)",
                         (oracle_a, oracle_b)).fetchall()
    if len(norms) < 2:
        return 0
    shared = conn.execute("""
        SELECT coalesce(sum(t.idf * t.idf), 0) AS s
        FROM card_tags ca
        JOIN card_tags cb ON cb.tag = ca.tag AND cb.oracle_id = %s
        JOIN tags t ON t.tag = ca.tag
        WHERE ca.oracle_id = %s""", (oracle_b, oracle_a)).fetchone()["s"]
    return concept_display(shared / (norms[0]["norm"] * norms[1]["norm"]))


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

def sideways(layout, type_line):
    #battles and split cards are printed sideways, so their pictures start
    #pre-rotated readable. battles dont get their own layout from scryfall
    #(theyre transform cards), but Battle always leads the type line
    return layout == "split" or "Battle" in (type_line or "")


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
    f["cmdr"] = request.args.get("cmdr") == "1"
    f["gc"] = request.args.get("gc") == "1"
    #flipped from the launch version: most visitors are commander players, so
    #cards that arent legal stay hidden unless this asks for them. old shared
    #links with legal=1 wanted exactly what the default now does, so they
    #still mean the same thing
    f["illegal"] = request.args.get("illegal") == "1"
    return f


def read_number(s):
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def read_min():
    #minimum match percent, in calibrated display units. 80 by default: the
    #map pins the model's real quality boundary to 80, so this keeps exactly
    #the set of cards the old raw-90 cutoff kept - the number just finally
    #means what it says, on both axes. under a price sort the percent is the
    #only quality gate left, so cheap half matches sorting to the top makes
    #the site look broken. everything below the line still exists, it just
    #pages in after the "show weaker matches" button instead of being thrown
    #away
    try:
        m = int(request.args.get("min", 80))
    except ValueError:
        m = 80
    return max(0, min(m, 100))


def read_sort():
    s = request.args.get("sort", "")
    if s not in ("cheap", "pricey"):
        s = "best"
    return s


def read_blend():
    #the slider's detent, 0 (pure mechanics) through 6 (pure concepts)
    try:
        b = int(request.args.get("blend", 0))
    except ValueError:
        b = 0
    return max(0, min(b, len(BLEND_WEIGHTS) - 1))


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
    if filters["cmdr"]:
        #commander targets. both words have to be in the type line
        #("Legendary Creature - Elf Warrior"), matching them separately also
        #catches things like "Legendary Enchantment Creature"
        where += " AND c.type_line ILIKE %s AND c.type_line ILIKE %s"
        params.append("%Legendary%")
        params.append("%Creature%")
    if filters["gc"]:
        where += " AND NOT c.game_changer"
    if not filters["illegal"]:
        where += " AND c.legal_commander"
    return where, params


def find_similar(oracle_id, picked, filters, min_pct, sort, offset=0, how_many=20, weak=False, blend=0.0):
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
        #pair's calibrated similarity, so that's what decides which side it
        #lands on
        strong = []
        weak_tier = []
        for entry in ranked:
            if mech_display(entry[1][0][1]) >= min_pct:
                strong.append(entry)
            else:
                weak_tier.append(entry)

        #the concepts side of the slider. two rules keep it honest: a card
        #shows if it passes EITHER axis's own gate (a concept match can join
        #the strong tier, never sneak past its own bar), and the blend only
        #decides ORDER - every displayed number stays what its own axis says
        concept_raw = {}
        if blend > 0:
            #every card sharing a tag with the anchor that clears the
            #concept gate, through the same filters as everything else
            rows = conn.execute("""
                WITH anchor AS (
                    SELECT ct.tag, t.idf FROM card_tags ct
                    JOIN tags t ON t.tag = ct.tag
                    WHERE ct.oracle_id = %s
                )
                SELECT ct.oracle_id, c.price_usd,
                       sum(a.idf * a.idf) / (na.norm * nc.norm) AS raw
                FROM card_tags ct
                JOIN anchor a ON a.tag = ct.tag
                JOIN cards c ON c.oracle_id = ct.oracle_id
                JOIN card_tag_norms nc ON nc.oracle_id = ct.oracle_id
                JOIN card_tag_norms na ON na.oracle_id = %s
                WHERE ct.oracle_id <> %s""" + where + """
                GROUP BY ct.oracle_id, c.price_usd, na.norm, nc.norm
                HAVING sum(a.idf * a.idf) / (na.norm * nc.norm) >= %s
                ORDER BY raw DESC
                LIMIT 300
            """, [oracle_id, oracle_id, oracle_id] + fparams + [concept_raw_gate(MIN_CONCEPT)]).fetchall()
            gated = {}
            for r in rows:
                gated[r["oracle_id"]] = r["raw"]
                prices.setdefault(r["oracle_id"], r["price_usd"])

            #the mechanical results need their concept scores too (below the
            #gate is fine here, they already earned their spot), so the blend
            #can weigh every card on both axes
            ids = [oid for oid, pairs in strong if oid not in gated]
            if ids:
                for r in conn.execute("""
                    WITH anchor AS (
                        SELECT ct.tag, t.idf FROM card_tags ct
                        JOIN tags t ON t.tag = ct.tag
                        WHERE ct.oracle_id = %s
                    )
                    SELECT ct.oracle_id,
                           sum(a.idf * a.idf) / (na.norm * nc.norm) AS raw
                    FROM card_tags ct
                    JOIN anchor a ON a.tag = ct.tag
                    JOIN card_tag_norms nc ON nc.oracle_id = ct.oracle_id
                    JOIN card_tag_norms na ON na.oracle_id = %s
                    WHERE ct.oracle_id = ANY(%s)
                    GROUP BY ct.oracle_id, na.norm, nc.norm
                """, (oracle_id, oracle_id, ids)).fetchall():
                    concept_raw[r["oracle_id"]] = r["raw"]
            concept_raw.update(gated)

            #concept-gated cards join the strong tier: out of the weak tier
            #if they were there, from nowhere if the lines never matched at
            #all (those carry no pairs and show their concept score instead)
            have = {oid for oid, pairs in strong}
            still_weak = []
            for oid, pairs in weak_tier:
                if oid in gated and oid not in have:
                    strong.append((oid, pairs))
                    have.add(oid)
                else:
                    still_weak.append((oid, pairs))
            weak_tier = still_weak
            for oid in gated:
                if oid not in have:
                    strong.append((oid, []))

            def blended(entry):
                oid, pairs = entry
                mech = mech_display(pairs[0][1]) if pairs else 0
                return (1 - blend) * mech + blend * concept_display(concept_raw.get(oid, 0.0))
            strong.sort(key=blended, reverse=True)
            #the weak tier gets the same lens, so its numbers read in order too
            weak_tier.sort(key=blended, reverse=True)

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

        #the tags each page card shares with the anchor, rarest first - the
        #chips that explain why the concepts side ranked a card where it did
        chips = {}
        if blend > 0 and ids:
            for r in conn.execute("""
                SELECT ct.oracle_id, ct.tag FROM card_tags ct
                JOIN card_tags a ON a.tag = ct.tag AND a.oracle_id = %s
                JOIN tags t ON t.tag = ct.tag
                WHERE ct.oracle_id = ANY(%s)
                ORDER BY t.idf DESC
            """, (oracle_id, ids)).fetchall():
                chips.setdefault(r["oracle_id"], []).append(r["tag"])

    results = []
    for oid, pairs in page:
        c = info[oid]
        concept_pct = concept_display(concept_raw[oid]) if oid in concept_raw else 0
        more = []
        if pairs:
            score, sim, our_line, their_line = pairs[0]
            mech_pct = mech_display(sim)
            concept_only = False
            #the extra pairs behind the "+n more matching lines" hint. pairs that
            #reuse a line already shown get skipped, so the count means genuinely
            #different abilities matched, not the same ability matching twice
            used_ours = [our_line]
            used_theirs = [their_line]
            for p in pairs[1:]:
                if p[2] in used_ours or p[3] in used_theirs:
                    continue
                used_ours.append(p[2])
                used_theirs.append(p[3])
                more.append('"' + p[3] + '" (' + str(mech_display(p[1])) + '%) matches your "' + p[2] + '"')
        else:
            #a pure concept match: no line of rules text got it here
            our_line = ""
            their_line = ""
            mech_pct = 0
            concept_only = True
        #the badge answers "how good a match, given where the slider is":
        #the pure mechanical percent at rest, and once the slider moves, the
        #same blend the ordering uses - so the list always reads in
        #descending order of the number shown. the two ingredients ride in
        #the tooltip
        if blend > 0:
            percent = int(round((1 - blend) * mech_pct + blend * concept_pct))
        else:
            percent = mech_pct
        price = ""
        if c["price_usd"] is not None:
            price = "$" + str(c["price_usd"])
        results.append({
            "oracle_id": str(oid),  #the report flag needs to say which card it's flagging
            "name": c["name"],
            "mana_cost": c["mana_cost"],
            "type_line": c["type_line"],
            "image": c["image"],
            "image_back": c["image_back"] or "",
            "sideways": sideways(c["layout"], c["type_line"]),
            "flip": c["layout"] == "flip",
            "scryfall_uri": c["scryfall_uri"],
            "percent": percent,
            "blended": blend > 0,
            "mech_pct": mech_pct,
            "concept_only": concept_only,
            "concept_pct": concept_pct,
            "concept_tags": ", ".join(chips.get(oid, [])[:3]),
            "our_line": our_line,
            "their_line": their_line,
            "price": price,
            "more_count": len(more),
            "more_text": "\n".join(more),
        })
    return results, has_more, weak_count


@app.route("/")
def home():
    #the landing page. search moved to /search when this page arrived, but
    #the launch thread links look like /?q=..., so anything with a query
    #gets forwarded there with its whole query string intact
    if request.args.get("q"):
        return redirect("/search?" + request.query_string.decode(), code=301)
    return render_template("home.html")


@app.route("/search")
def search():
    query = request.args.get("q", "")
    if not query:
        return redirect("/")

    card = find_card(query)
    if card is None:
        return render_template("search.html", query=query, not_found=True)

    #the searched card's picture gets the same rotate/flip/transform frame
    #as everything else, the template just needs the two flags
    card = dict(card)
    card["sideways"] = sideways(card["layout"], card["type_line"])
    card["flip"] = card["layout"] == "flip"

    filters = read_filters()
    min_pct = read_min()
    sort = read_sort()
    blend = read_blend()
    card_lines, picked = build_lines(card, read_picked())

    results, has_more, weak_count = find_similar(card["oracle_id"], picked, filters, min_pct, sort,
                                                 blend=BLEND_WEIGHTS[blend])
    return render_template("search.html", query=query, card=card, card_lines=card_lines,
                           picked_count=len(picked), results=results, has_more=has_more,
                           weak_count=weak_count, min_pct=min_pct, blend=blend, types=CARD_TYPES)


def read_unique():
    #the u param on /unique/cards, how unique a card has to be before the
    #shuffle is allowed to deal it. same clamping story as read_min
    try:
        u = int(request.args.get("u", UNIQUE_DEFAULT))
    except ValueError:
        u = UNIQUE_DEFAULT
    return max(0, min(u, 100))


@app.route("/unique")
def unique():
    return render_template("unique.html", types=CARD_TYPES, unique_default=UNIQUE_DEFAULT)


def card_json(c):
    #one dealt (or revisited) card the way the /unique frontend wants it.
    #layout and image_back are what let the page pre-rotate sideways cards
    #and offer the turn-over button
    price = ""
    if c["price_usd"] is not None:
        price = "$" + str(c["price_usd"])
    return {
        "oracle_id": str(c["oracle_id"]),
        "name": c["name"],
        "mana_cost": c["mana_cost"],
        "type_line": c["type_line"],
        "image": c["image"],
        "image_back": c["image_back"] or "",
        "sideways": sideways(c["layout"], c["type_line"]),
        "flip": c["layout"] == "flip",
        "scryfall_uri": c["scryfall_uri"],
        "price": price,
        "percent": int(round((c["uniqueness"] or 0) * 100)),
        "unique_line": c["unique_line"] or "",
    }


@app.route("/unique/cards", methods=["POST"])
def unique_cards():
    #deals UNIQUE_PAGE random cards whose uniqueness clears the bar, skipping
    #ones the browser has already been shown. the filters ride in on the query
    #string like everywhere else, but the seen list arrives as a json body
    #because after enough dealing it outgrows what a url can carry. its a
    #random draw from everything that qualifies, not the top of a ranking, so
    #every press feels like a fresh pack
    filters = read_filters()
    u = read_unique()
    body = request.get_json(silent=True) or {}
    seen = []
    for s in body.get("seen", []):
        #only real uuids get through to the query, anything else in
        #localStorage was not put there by us
        try:
            seen.append(str(uuid.UUID(str(s))))
        except ValueError:
            pass

    where, fparams = filter_sql(filters)
    #the -0.5 makes the sql cutoff agree with the rounded percent the page
    #shows: a card displayed as exactly u% still qualifies at u
    cond = """
        FROM cards c
        WHERE c.uniqueness * 100.0 >= %s - 0.5
          AND NOT (c.oracle_id = ANY(%s::uuid[]))""" + where
    params = [u, seen] + fparams
    with pool.connection() as conn:
        remaining = conn.execute("SELECT count(*) AS n" + cond, params).fetchone()["n"]
        rows = conn.execute("SELECT " + CARD_FIELDS + ", uniqueness, unique_line" + cond +
                            " ORDER BY random() LIMIT %s", params + [UNIQUE_PAGE]).fetchall()

    cards = []
    for c in rows:
        cards.append(card_json(c))
    #remaining counts whats left AFTER this deal, so the frontend knows when
    #the well is dry without another request
    return {"cards": cards, "remaining": remaining - len(cards)}


@app.route("/unique/card")
def unique_card():
    #one card the browser has already been dealt, looked up by id for the
    #back/forward history arrows on /unique. same shape as a fresh deal so
    #the frontend renders both identically. cards can vanish from the
    #database between visits (scryfall drops them, filters tighten), so
    #null just means "this history entry died"
    try:
        oid = str(uuid.UUID(request.args.get("id", "")))
    except ValueError:
        return {"card": None}
    with pool.connection() as conn:
        c = conn.execute("SELECT " + CARD_FIELDS + ", uniqueness, unique_line FROM cards WHERE oracle_id = %s",
                         (oid,)).fetchone()
    if c is None:
        return {"card": None}
    return {"card": card_json(c)}


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
    results, has_more, weak_count = find_similar(card["oracle_id"], picked, read_filters(), read_min(), read_sort(), offset,
                                                 weak=weak, blend=BLEND_WEIGHTS[read_blend()])
    return {"results": results, "has_more": has_more, "weak_count": weak_count}


#---- user feedback: "a card is missing" / "this card shouldn't be here" ----

def client_ip():
    #railway sits behind a proxy, so the visitor's real address is the first
    #entry of the forwarded list, not remote_addr
    fwd = request.headers.get("X-Forwarded-For", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.remote_addr or ""


def best_sim(conn, anchor_id, other_id, picked):
    #the calibrated percent the results page prints next to the other card.
    #the winning pair is chosen by WEIGHTED similarity, exactly like the
    #ranking (a bare max would disagree with the page: boots and greaves
    #share near-identical Equip lines at raw .99 that the idf weighting
    #buries), and the number returned is that pair's real similarity. None
    #means the other card has no searchable lines at all
    sql = """
        SELECT 1 - (a.embedding <=> b.embedding) AS sim,
               coalesce(s.count, 1) AS count
        FROM lines a
        JOIN lines b ON b.oracle_id = %s
        LEFT JOIN line_stats s ON s.line_text = a.line_text
        WHERE a.oracle_id = %s
    """
    params = [other_id, anchor_id]
    if picked:
        sql += " AND a.line_text = ANY(%s)"
        params.append(list(picked))
    best = None
    for r in conn.execute(sql, params):
        weighted = line_weight(r["count"]) * r["sim"]
        if best is None or weighted > best[0]:
            best = (weighted, r["sim"])
    if best is None:
        return None
    return mech_display(best[1])


def filter_reasons(card, filters):
    #why the current filters hide this card, said in the user's language.
    #mirrors filter_sql condition by condition; an empty list means the
    #filters let the card through and something else explains its absence
    reasons = []
    if filters["colors"]:
        for letter in card["color_identity"]:
            if letter not in filters["colors"]:
                reasons.append("its color identity (" + card["color_identity"] + ") doesn't fit the colors you picked")
                break
    if (filters["pmin"] is not None or filters["pmax"] is not None) and card["price_usd"] is None:
        reasons.append("it has no listed price, and any price filter hides unpriced cards")
    elif filters["pmin"] is not None and float(card["price_usd"]) < filters["pmin"]:
        reasons.append("its price ($" + str(card["price_usd"]) + ") is under your minimum")
    elif filters["pmax"] is not None and float(card["price_usd"]) > filters["pmax"]:
        reasons.append("its price ($" + str(card["price_usd"]) + ") is over your maximum")
    if filters["mvmin"] is not None and float(card["cmc"]) < filters["mvmin"]:
        reasons.append("its mana value (" + str(card["cmc"]) + ") is under your minimum")
    if filters["mvmax"] is not None and float(card["cmc"]) > filters["mvmax"]:
        reasons.append("its mana value (" + str(card["cmc"]) + ") is over your maximum")
    if filters["type"] and filters["type"].lower() not in (card["type_line"] or "").lower():
        reasons.append("its type line doesn't include " + filters["type"])
    if filters["cmdr"]:
        tl = (card["type_line"] or "").lower()
        if "legendary" not in tl or "creature" not in tl:
            reasons.append("it can't be a commander and \"commanders only\" is on")
    if filters["gc"] and card["game_changer"]:
        reasons.append("it's a game changer and \"hide game changers\" is on")
    if not filters["illegal"] and not card["legal_commander"]:
        reasons.append("it isn't commander-legal, and those stay hidden unless \"include illegal\" is ticked")
    return reasons


@app.route("/feedback", methods=["POST"])
def feedback():
    #the report bar on the results page posts here. the page's whole query
    #string rides along exactly like /more does, so the report is judged
    #against the same anchor, picked lines, filters and cutoff the user was
    #actually looking at.
    #
    #two kinds: 'missing' (a good card that should have been in the results,
    #a future pairs.md entry) and 'misplaced' (a bad card that shouldn't be
    #here, with the user's reason in their own words, a future triplets.md
    #negative). nobody is asked to name a replacement card, most players
    #couldn't quote one on the spot. missing reports get diagnosed before
    #anything is stored: when a filter is what's hiding the card, the user
    #learns that on the spot and the review queue never hears about it,
    #because that's not the model's fault
    body = request.get_json(silent=True) or {}
    kind = body.get("kind", "")
    if kind not in ("missing", "misplaced"):
        return {"ok": False, "stored": False, "msg": "That report didn't make sense to the server, sorry."}

    card = find_card(request.args.get("q", ""))
    if card is None:
        return {"ok": False, "stored": False, "msg": "Lost track of which card you searched, try reloading the page."}

    reason = str(body.get("reason", "")).strip()[:500]

    filters = read_filters()
    min_pct = read_min()
    _, picked = build_lines(card, read_picked())

    with pool.connection() as conn:
        #a gentle lid, there's no login so this is all the abuse control there is
        ip = client_ip()
        recent = conn.execute("SELECT count(*) AS n FROM feedback WHERE ip = %s AND created_at > now() - interval '1 hour'",
                              (ip,)).fetchone()["n"]
        if recent >= 20:
            return {"ok": False, "stored": False, "msg": "That's a lot of reports for one hour. Thank you, but please come back later."}

        #which model's numbers this report is about, straight from the ingest's bookkeeping
        row = conn.execute("SELECT value FROM meta WHERE key = 'embed_model'").fetchone()
        model = row["value"] if row else ""
        snap = dict(filters)
        snap["min"] = min_pct
        snap["sort"] = read_sort()
        #the slider position changes what the numbers the user saw MEANT
        #(blended past detent 0), so it rides in the snapshot too
        blend = read_blend()
        snap["blend"] = blend
        #scale marker: reports from before 2026-07-15 stored raw-cosine
        #percents, everything after stores calibrated display percents
        snap["cal"] = 1

        if kind == "missing":
            expected_name = str(body.get("expected", "")).strip()[:200]
            expected = find_card(expected_name) if expected_name else None
            if expected is None:
                return {"ok": False, "stored": False, "msg": 'Couldn\'t find a card called "' + expected_name + '", check the spelling?'}
            if expected["oracle_id"] == card["oracle_id"]:
                return {"ok": False, "stored": False, "msg": expected["name"] + " is the card you searched for."}
            expected_pct = best_sim(conn, card["oracle_id"], expected["oracle_id"], picked)
            if expected_pct is None:
                return {"ok": False, "stored": False, "msg": expected["name"] + " has no rules text the matcher can search, so it can never appear."}
            full = conn.execute("""SELECT color_identity, price_usd, cmc, type_line, game_changer, legal_commander
                                   FROM cards WHERE oracle_id = %s""", (expected["oracle_id"],)).fetchone()
            reasons = filter_reasons(full, filters)
            if reasons:
                return {"ok": True, "stored": False,
                        "msg": expected["name"] + " matches at " + str(expected_pct) + "%, but your filters hide it: " + "; ".join(reasons) + "."}
            if expected_pct >= min_pct:
                return {"ok": True, "stored": False,
                        "msg": expected["name"] + " is in the results at " + str(expected_pct) + "%, it may just be further down the list."}
            if blend > 0:
                #with the slider away from mechanics the card may already be
                #on the page as a concept match, which is not a model gap
                cpct = concept_between(conn, card["oracle_id"], expected["oracle_id"])
                snap["concept_pct"] = cpct
                if cpct >= MIN_CONCEPT:
                    return {"ok": True, "stored": False,
                            "msg": expected["name"] + " already shows as a concept match at " + str(cpct) +
                                   "% with your slider, it may just be further down the list."}
            #a real gap: nothing hides the card, the model just scores it under the cutoff
            conn.execute("""INSERT INTO feedback (kind, anchor_id, anchor_name, expected_id, expected_name,
                                                  expected_pct, reason, picked_lines, filters, embed_model, ip)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                         (kind, card["oracle_id"], card["name"], expected["oracle_id"], expected["name"],
                          expected_pct, reason, "\n".join(picked), json.dumps(snap), model, ip))
            return {"ok": True, "stored": True,
                    "msg": "Logged. " + expected["name"] + " only scores " + str(expected_pct) + "% against the " + str(min_pct) +
                           "% cutoff (for now it's behind the weaker-matches button). Reports like this grade the next model."}

        #misplaced: the flagged card plus a few words on why it doesn't belong
        try:
            got_id = str(uuid.UUID(str(body.get("got_id", ""))))
        except ValueError:
            return {"ok": False, "stored": False, "msg": "Lost track of which result you flagged, try reloading the page."}
        got = conn.execute("SELECT oracle_id, name FROM cards WHERE oracle_id = %s", (got_id,)).fetchone()
        if got is None:
            return {"ok": False, "stored": False, "msg": "Lost track of which result you flagged, try reloading the page."}
        if not reason:
            return {"ok": False, "stored": False, "msg": "Say a few words about why it's a bad match first."}

        got_pct = best_sim(conn, card["oracle_id"], got["oracle_id"], picked)
        if blend > 0:
            #the badge the user flagged was blended, so keep the concept half
            #on file - the review needs it to route the report to an axis
            snap["concept_pct"] = concept_between(conn, card["oracle_id"], got["oracle_id"])
        conn.execute("""INSERT INTO feedback (kind, anchor_id, anchor_name, got_id, got_name,
                                              got_pct, reason, picked_lines, filters, embed_model, ip)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                     (kind, card["oracle_id"], card["name"], got["oracle_id"], got["name"],
                      got_pct, reason, "\n".join(picked), json.dumps(snap), model, ip))
        return {"ok": True, "stored": True,
                "msg": "Logged. " + got["name"] + " shows at " + str(got_pct) +
                       "% right now, and reports like this become the test cases the matcher is graded against."}


#---- the review side of the feedback loop ----

def admin_allowed():
    #no ADMIN_KEY in the environment means no admin pages anywhere, and a
    #wrong key 404s instead of 403 so the page doesn't admit it exists
    return ADMIN_KEY != "" and request.args.get("key", "") == ADMIN_KEY


def report_markdown(r, line_texts, n):
    #one accepted report in the shape pairs.md uses, ready to paste (the
    #separators mirror that file exactly). missing reports become should-match
    #entries (the anchor and the good card), misplaced reports become
    #should-NOT entries (the anchor, the bad card and the user's reason).
    #promotion into triplets.md happens by hand at review time. the anchor
    #quotes only the picked lines when the report came from a line-picked
    #search
    def q(lines):
        if not lines:
            return "`(card no longer in the database)`"
        return " + ".join("`" + t + "`" for t in lines)

    if r["picked_lines"]:
        anchor_lines = r["picked_lines"].split("\n")
    else:
        anchor_lines = line_texts.get(r["anchor_id"], [])
    day = r["created_at"].strftime("%Y-%m-%d")

    #a report filed with the slider away from mechanics was judging blended
    #numbers, and probably belongs in axis2.md rather than pairs.md. the
    #stored pcts stay mechanical either way, this note carries the rest
    mode = ""
    try:
        snap = json.loads(r["filters"] or "{}")
    except ValueError:
        snap = {}
    if snap.get("blend"):
        try:
            at = str(int(BLEND_WEIGHTS[int(snap["blend"])] * 100)) + "% concepts"
        except (ValueError, IndexError):
            at = "detent " + str(snap["blend"])
        mode = "; slider at " + at + " (user saw blended numbers"
        if "concept_pct" in snap:
            mode += ", concept score " + str(snap["concept_pct"]) + "%"
        mode += ") - consider axis2.md"

    out = str(n) + ".\n"
    out += "    **Anchor:** " + r["anchor_name"] + " — " + q(anchor_lines) + "\n"
    if r["kind"] == "misplaced":
        out += "    **NOT:** " + (r["got_name"] or "?") + " — " + q(line_texts.get(r["got_id"], [])) + "\n"
        out += "    *user report " + day + "; the flagged card showed at " + str(r["got_pct"]) + "% mech" + mode + "; reason: " + r["reason"] + "*\n"
    else:
        out += "    **Match:** " + (r["expected_name"] or "?") + " — " + q(line_texts.get(r["expected_id"], [])) + "\n"
        note = "user report " + day + "; scored " + str(r["expected_pct"]) + "% mech against the cutoff" + mode
        if r["reason"]:
            note += "; reason: " + r["reason"]
        out += "    *" + note + "*\n"
    return out


@app.route("/admin")
def admin():
    if not admin_allowed():
        abort(404)
    with pool.connection() as conn:
        rows = conn.execute("""SELECT * FROM feedback WHERE status IN ('pending', 'accepted')
                               ORDER BY created_at DESC""").fetchall()
        #one round trip for every card picture and line text the page shows
        ids = set()
        for r in rows:
            ids.add(r["anchor_id"])
            if r["expected_id"]:
                ids.add(r["expected_id"])
            if r["got_id"]:
                ids.add(r["got_id"])
        info = {}
        line_texts = {}
        if ids:
            for c in conn.execute("SELECT oracle_id, name, image FROM cards WHERE oracle_id = ANY(%s)", (list(ids),)):
                info[c["oracle_id"]] = c
            for l in conn.execute("SELECT oracle_id, line_text FROM lines WHERE oracle_id = ANY(%s)", (list(ids),)):
                line_texts.setdefault(l["oracle_id"], []).append(l["line_text"])

    def card_bit(role, oid, name, pct):
        c = info.get(oid)
        return {"role": role, "name": name, "image": c["image"] if c else "", "pct": pct}

    pending = []
    accepted = []
    triplet_md = []
    pair_md = []
    for r in rows:
        cards = [card_bit("anchor (searched)", r["anchor_id"], r["anchor_name"], None)]
        if r["expected_id"]:
            cards.append(card_bit("should appear", r["expected_id"], r["expected_name"], r["expected_pct"]))
        if r["got_id"]:
            cards.append(card_bit("shouldn't be here", r["got_id"], r["got_name"], r["got_pct"]))
        view = {
            "id": r["id"], "kind": r["kind"], "cards": cards, "reason": r["reason"],
            "created": r["created_at"].strftime("%Y-%m-%d %H:%M"),
            "picked": r["picked_lines"].replace("\n", "  |  "),
            "filters": r["filters"], "model": r["embed_model"], "ip": r["ip"],
        }
        if r["status"] == "pending":
            pending.append(view)
        else:
            accepted.append(view)
            if r["kind"] == "misplaced":
                triplet_md.append(report_markdown(r, line_texts, len(triplet_md) + 1))
            else:
                pair_md.append(report_markdown(r, line_texts, len(pair_md) + 1))

    return render_template("admin.html", key=ADMIN_KEY, pending=pending, accepted=accepted,
                           triplet_md="\n".join(triplet_md), pair_md="\n".join(pair_md))


@app.route("/admin/act", methods=["POST"])
def admin_act():
    if not admin_allowed():
        abort(404)
    try:
        fid = int(request.form.get("id", ""))
    except ValueError:
        abort(400)
    action = request.form.get("action", "")
    #archived is where accepted reports go once they've been copied into the eval files
    if action not in ("accepted", "rejected", "archived", "pending"):
        abort(400)
    with pool.connection() as conn:
        conn.execute("UPDATE feedback SET status = %s WHERE id = %s", (action, fid))
    return redirect("/admin?key=" + ADMIN_KEY)


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
