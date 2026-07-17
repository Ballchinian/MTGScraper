#the actual website. the old version loaded a 90mb embedding matrix into
#memory at startup, this one just asks postgres. the embeddings live in the
#database and pgvector does the similarity math right where the data is, so
#this process stays tiny and never touches torch

import re
import os
import math
import time
import uuid
import json
import hashlib
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, render_template, request, redirect, abort, make_response, url_for, Response
from flask_compress import Compress
from werkzeug.middleware.proxy_fix import ProxyFix

from db import pool
from prefix_words import PREFIX_WORDS

app = Flask(__name__)

#railway terminates tls one proxy in front of this app, so without this
#flask believes every request was plain http on an internal hostname. the
#canonical and og:url tags embed request.url_root, and those must say https
#on the real domain or google treats every page as its http twin
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

#gzip for every text response (html, json, css, js). the search page and the
#/more payloads are prose-heavy and shrink several times over
Compress(app)

#static files may cache for a year because static_url below stamps a content
#hash onto every url the templates emit: changing a file changes its url, so
#a stale cache can never serve an old stylesheet against a new page
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 60 * 60 * 24 * 365

_static_hash = {}


@app.template_global()
def static_url(filename):
    v = _static_hash.get(filename)
    if v is None:
        with open(os.path.join(app.static_folder, filename), "rb") as f:
            v = hashlib.md5(f.read()).hexdigest()[:8]
        _static_hash[filename] = v
    return url_for("static", filename=filename) + "?v=" + v

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
CARD_FIELDS = "oracle_id, name, mana_cost, type_line, oracle_text, image, scryfall_uri, price_usd, price_eur, layout, image_back, edhrec_rank"

#the choices in the type filter dropdown. also acts as a whitelist so
#nothing weird from the url ends up inside a LIKE pattern
CARD_TYPES = ["Creature", "Instant", "Sorcery", "Artifact", "Enchantment", "Planeswalker", "Battle", "Land"]

#the /unique page deals this many cards per request. one at a time, its the
#counterpart to scryfall's random card button
UNIQUE_PAGE = 1

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


#axis 2: how conceptually close two cards are, scored from the community
#tags the ingest bakes into card_tags/tags. the raw cosine lives in a
#compressed band, this map turns it into the percent the site shows, and
#the gate is written in displayed units on purpose.
#
#this and MECH_CALIBRATION below are SEEDS: the ingest writes the real maps
#into meta next to the model name they're anchored to, and load_calibration
#(further down, once both are defined) makes the database's word win. these
#only hold until the first ingest run against a database, and a model swap
#carries its new map along with its new vectors automatically
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
#one. raw cosine is arbitrary per model, so the displayed percent is pinned
#to judged pairs instead. the map is anchored to the tuned embeddinggemma
#the ingest embeds with, and the full story of the anchors lives next to
#EMBED_MODEL in ingest/update.py, which is the source of truth that lands
#in meta. this copy is the seed for databases the ingest hasn't touched yet
MECH_CALIBRATION = [(0.0, 0), (0.50, 30), (0.70, 45), (0.85, 65), (0.895, 80), (0.97, 92), (1.0, 100)]


def load_calibration():
    #the maps the ingest wrote into meta replace the seeds above, so the
    #percents the site shows always belong to the model that made the
    #vectors. a database the ingest has never run against has no meta rows
    #(maybe no meta table), then the seeds hold
    global CALIBRATION, MECH_CALIBRATION
    try:
        with pool.connection() as conn:
            for key in ("concept_calibration", "mech_calibration"):
                row = conn.execute("SELECT value FROM meta WHERE key = %s", (key,)).fetchone()
                if row:
                    pts = [(float(x), float(y)) for x, y in json.loads(row["value"])]
                    if key == "concept_calibration":
                        CALIBRATION = pts
                    else:
                        MECH_CALIBRATION = pts
    except Exception:
        pass


load_calibration()


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
    #one query instead of up to four round trips: every way a name can match
    #(exact, starts-with, anywhere, trigram-fuzzy) becomes a tier and the
    #best-tiered card wins. inside the substring tiers alphabetical order
    #decides (trigram closeness would favor short names, "delver" must find
    #Delver of Secrets, not Delver's Torch), the fuzzy tier goes closest
    #first - its alphabetical CASE key is NULL, which sorts after every real
    #name. the % operator means "similar enough to bother" (so garbage
    #queries still return nothing) and <-> sorts by closest. %% because
    #psycopg uses % for parameters
    q = query.strip()
    with pool.connection() as conn:
        return conn.execute("""
            SELECT """ + CARD_FIELDS + """,
                   CASE WHEN lower(name) = lower(%s) THEN 0
                        WHEN name ILIKE %s THEN 1
                        WHEN name ILIKE %s THEN 2
                        ELSE 3 END AS tier
            FROM cards
            WHERE lower(name) = lower(%s) OR name ILIKE %s OR name ILIKE %s OR name %% %s
            ORDER BY tier, CASE WHEN name ILIKE %s THEN name END, name <-> %s, name
            LIMIT 1
        """, (q, q + "%", "%" + q + "%", q, q + "%", "%" + q + "%", q, "%" + q + "%", q)).fetchone()


#all the url params past q and offset live in these little readers. home()
#and /more both use them, so the load more button sees exactly the same
#filters and picked lines as the page it's glued onto

def rank_label(rank):
    #edhrec's popularity rank next to the price, 1 being the most played card
    #in the format. plenty of cards are unranked (nobody plays them, or
    #they're too new to have a rank yet), those just show nothing
    return "#" + str(rank) if rank is not None else ""


def price_label(c, currency):
    #the price string under a card, in whichever currency the toggle picked.
    #empty when the card has no price in that currency
    p = c["price_usd"] if currency == "usd" else c["price_eur"]
    if p is None:
        return ""
    return ("$" if currency == "usd" else "€") + str(p)


def sideways(layout, type_line):
    #battles and split cards are printed sideways, so their frames offer the
    #rotate button. battles dont get their own layout from scryfall (theyre
    #transform cards), but Battle always leads the type line
    return layout == "split" or "Battle" in (type_line or "")


#---- the filter box compiler ----
#scryfall style syntax with parentheses, or/and and negation, compiled into
#one sql condition over the cards table:
#  o:word / o:"a phrase"   rules text contains it       t:creature   type line word
#  id:wug                  identity fits inside, like deckbuilding. id:c colorless
#  otag:removal            the community tag, straight from our own daily
#                          mirror of scryfall's tagger data
#  is:dfc is:split         card layout. is:gamechanger for the edh watchlist
#  f:commander             legal in commander. banned:commander for the reverse
#  usd>=1 eur<5 mv=2       price and mana value, cmc works for mv, eur too,
#                          and a bare price<5 follows the currency toggle
#  - before anything negates it, words side by side mean and, "or" means or,
#  parens group: (o:draw or o:scry) -t:creature usd<5
#anything unrecognisable is skipped, a typo never breaks the search

FQ_TOKEN = re.compile(r"""
    (?P<paren>[()])
  | (?P<kw>and|or)(?=[\s()]|$)
  | (?P<neg>-)
  | (?P<cfield>usd|price|eur|mv|cmc)\s*(?P<op>>=|<=|=|>|<)\s*(?P<num>\d+(?:\.\d+)?)
  | (?P<key>[a-z]+):(?:"(?P<qval>[^"]*)"|(?P<val>[^\s()]+))
  | (?P<junk>[^\s()]+)
""", re.IGNORECASE | re.VERBOSE)


def fq_term(key, value):
    #one key:value into (sql, params), or None for keys we don't speak (yet)
    key = key.lower()
    if key in ("o", "oracle"):
        return "c.oracle_text ILIKE %s", ["%" + value + "%"]
    if key in ("t", "type"):
        return "c.type_line ILIKE %s", ["%" + value + "%"]
    if key == "id":
        letters = "".join(ch for ch in value.upper() if ch in "WUBRG")
        if letters:
            return "c.color_identity ~ %s", ["^[" + letters + "]*$"]
        if "C" in value.upper():
            return "c.color_identity = ''", []
        return None
    if key in ("is", "layout"):
        #only the layout side of scryfall's is:. an unknown value (is:permanent,
        #is:reserved...) falls through to None and is skipped, never matched to
        #an empty set, same as any other key we don't speak
        v = value.lower()
        if v == "dfc":
            #a real two-sided card, the layouts that print a back face
            return "c.layout IN ('transform', 'modal_dfc', 'meld')", []
        if v in ("mdfc", "modal"):
            return "c.layout = 'modal_dfc'", []
        if v == "gamechanger":
            return "c.game_changer = true", []
        #the layouts that actually occur in the cards table. battle and token
        #are deliberately absent: battles arrive from scryfall as transform
        #cards and tokens never enter the database, so both would silently
        #match nothing instead of being skipped like any other unknown value
        if v in ("normal", "split", "flip", "transform", "modal_dfc", "meld",
                 "leveler", "class", "case", "saga", "adventure", "mutate",
                 "prototype", "prepare"):
            return "c.layout = %s", [v]
        return None
    if key in ("f", "format", "legal", "banned"):
        #commander is the only format whose legality we track
        if value.lower() in ("commander", "edh"):
            return "c.legal_commander = " + ("false" if key == "banned" else "true"), []
        return None
    if key in ("otag", "tag", "oracletag", "function"):
        #tagger tags form a hierarchy and the taggings sit on the leaves
        #(otag:removal itself tags nothing, spot-removal and friends do), so
        #the match walks the family tree downward, same as scryfall does
        return ("""EXISTS (SELECT 1 FROM card_tags ct WHERE ct.oracle_id = c.oracle_id AND ct.tag IN (
                     WITH RECURSIVE fam AS (
                         SELECT tag FROM tags WHERE tag = %s
                         UNION
                         SELECT t.tag FROM tags t JOIN fam f ON f.tag = ANY(t.parents)
                     ) SELECT tag FROM fam))""", [value.lower()])
    return None


def compile_fq(fq, currency="usd"):
    #tokenize then a little recursive descent. fail-soft on purpose: skipped
    #tokens and unbalanced parens degrade to a smaller filter, never a 500
    tokens = []
    for m in FQ_TOKEN.finditer(fq or ""):
        if m.group("paren"):
            tokens.append((m.group("paren"), None))
        elif m.group("kw"):
            tokens.append((m.group("kw").lower(), None))
        elif m.group("neg"):
            tokens.append(("-", None))
        elif m.group("cfield"):
            cf = m.group("cfield").lower()
            #usd and eur always mean themselves, the bare word price follows
            #the currency toggle
            cur_col = "c.price_usd" if currency == "usd" else "c.price_eur"
            col = {"usd": "c.price_usd", "eur": "c.price_eur", "price": cur_col}.get(cf, "c.cmc")
            tokens.append(("term", (col + " " + m.group("op") + " %s", [float(m.group("num"))])))
        elif m.group("key"):
            value = m.group("qval") if m.group("qval") is not None else m.group("val")
            tokens.append(("term", fq_term(m.group("key"), value)))
        #bare words fall through and are ignored
    pos = [0]

    def peek():
        return tokens[pos[0]][0] if pos[0] < len(tokens) else None

    def take():
        t = tokens[pos[0]]
        pos[0] += 1
        return t

    def parse_unary():
        kind, payload = take()
        if kind == "-":
            inner = parse_unary()
            return ("NOT (" + inner[0] + ")", inner[1]) if inner else None
        if kind == "(":
            expr = parse_or()
            if peek() == ")":
                take()
            #keep the user's grouping in the sql, AND binds tighter than OR
            return ("(" + expr[0] + ")", expr[1]) if expr else None
        if kind == "term":
            return payload  #already (sql, params), or None for unknown keys
        return None  #stray ) or keyword, skip it

    def parse_and():
        parts = []
        while peek() not in (None, ")", "or"):
            if peek() == "and":
                take()
                continue
            unit = parse_unary()
            if unit:
                parts.append(unit)
        if not parts:
            return None
        sql = " AND ".join(p[0] for p in parts)
        return sql, [x for p in parts for x in p[1]]

    def parse_or():
        parts = []
        unit = parse_and()
        if unit:
            parts.append(unit)
        while peek() == "or":
            take()
            unit = parse_and()
            if unit:
                parts.append(unit)
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        sql = " OR ".join("(" + p[0] + ")" for p in parts)
        return sql, [x for p in parts for x in p[1]]

    parts = []
    while pos[0] < len(tokens):
        expr = parse_or()
        if expr:
            parts.append(expr)
        elif peek() is not None:
            take()  #stray token, keep moving
    if not parts:
        return None, []
    sql = " AND ".join("(" + p[0] + ")" for p in parts)
    return sql, [x for p in parts for x in p[1]]


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
    #which currency the price bounds (and the filter box's bare price) mean
    f["cur"] = read_currency()
    #the filter box rides in as fq, compiled here so every page that reads
    #filters understands it. it stacks with the widgets (both apply)
    f["fq_sql"], f["fq_params"] = compile_fq(request.args.get("fq", ""), f["cur"])
    return f


def read_number(s):
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def default_min(blend):
    #80 at both ENDS of the slider: pure mechanics pins the model's real
    #quality boundary there (same set of cards the old raw-90 cutoff kept),
    #and pure concepts shows the calibrated concept score, where good matches
    #also read 80+. the mixed detents show an average of two axes though, and
    #averages rarely reach 80, so they relax to 70
    return 70 if 0 < blend < len(BLEND_WEIGHTS) - 1 else 80


def read_min(blend=0):
    #minimum match percent, in calibrated display units. the relaxing above is
    #only the DEFAULT: an explicit min in the url always wins, and the page
    #offers a way back to the default whenever one is overriding it.
    #everything below the line still exists either way, it just pages in
    #behind the "show weaker matches" button instead of being thrown away
    try:
        m = int(request.args.get("min", ""))
    except ValueError:
        m = default_min(blend)
    return max(0, min(m, 100))


def read_sort():
    s = request.args.get("sort", "")
    if s not in ("cheap", "pricey", "played", "obscure"):
        s = "best"
    return s


def read_currency():
    #usd or eur, for every price the site shows, filters and sorts on. the
    #url wins, then the remembered cookie, then dollars. scryfall only
    #prices those two currencies, so thats the whole menu
    cur = request.args.get("cur")
    if cur is None:
        cur = request.cookies.get("cur", "usd")
    return cur if cur in ("usd", "eur") else "usd"


@app.after_request
def remember_currency(resp):
    #any request that names a currency makes it the remembered one, so the
    #toggle sticks no matter which page it was flipped on (/search submits
    #its form, /unique deals and trail-walks through fetch)
    cur = request.args.get("cur")
    if cur in ("usd", "eur"):
        resp.set_cookie("cur", cur, max_age=60 * 60 * 24 * 365, samesite="Lax")
    return resp


def read_blend():
    #the slider's detent, 0 (pure rules text) through 4 (pure concepts).
    #the url wins, then the remembered cookie, then 0. people who like the
    #slider somewhere tend to want it there tomorrow too, so the preference
    #survives closing the browser
    raw = request.args.get("blend")
    if raw is None:
        raw = request.cookies.get("blend", "0")
    try:
        b = int(raw)
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
    #the bounds compare in whichever currency the toggle shows, so what you
    #filter on is always the number printed under the cards
    pcol = "c.price_usd" if filters.get("cur", "usd") == "usd" else "c.price_eur"
    if filters["pmin"] is not None:
        where += " AND " + pcol + " >= %s"
        params.append(filters["pmin"])
    if filters["pmax"] is not None:
        #cards with no known price fail both comparisons, so any price filter
        #quietly drops them, which is what a budget search wants
        where += " AND " + pcol + " <= %s"
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
    if filters.get("fq_sql"):
        #the compiled filter box, one boolean expression over the cards row
        where += " AND (" + filters["fq_sql"] + ")"
        params.extend(filters["fq_params"])
    return where, params


def find_similar(oracle_id, picked, filters, min_pct, sort, offset=0, how_many=20, weak=False, blend=0.0, currency="usd"):
    #every candidate card keeps all its matching line pairs now instead of
    #just the best one, so results can show "+2 more matching lines".
    #
    #cards split into two tiers around min_pct: the strong ones are the real
    #results, the weak ones sit behind the "show weaker matches" button.
    #weak=True means the caller is paging through that second tier
    pairs_by_card = {}  #other card's oracle_id -> list of (weighted score, real similarity, our line, their line)
    prices = {}         #other card's oracle_id -> price in the chosen currency, for the price sorts
    ranks = {}          #other card's oracle_id -> edhrec rank, for the played sorts
    where, fparams = filter_sql(filters)
    #the column the price sorts read, matching the currency the page prints
    pcol = "c.price_usd" if currency == "usd" else "c.price_eur"
    #the query card's lines are already embedded in the database, so the
    #model never runs at search time. grab them with their idf counts in one
    #go, on a briefly borrowed connection
    with pool.connection() as conn:
        qlines = conn.execute("""
            SELECT l.line_text, l.embedding, coalesce(s.count, 1) AS count
            FROM lines l LEFT JOIN line_stats s ON s.line_text = l.line_text
            WHERE l.oracle_id = %s AND NOT l.whole
        """, (oracle_id,)).fetchall()

    #if the user picked lines on the page, only search with those
    if picked:
        chosen = []
        for ql in qlines:
            if ql["line_text"] in picked:
                chosen.append(ql)
        if chosen:
            qlines = chosen

    def hunt(ql):
        #one line's nearest neighbor walk through the hnsw index, ~10-20ms
        #where the exact scan it replaced measured 200-250ms. <=> is cosine
        #distance, so similarity is 1 minus it. grab way more than we need
        #since a bunch will get merged per card.
        #
        #one query per line on purpose. the obvious "one big query" (the
        #anchor's lines CROSS JOIN LATERAL the scan) was built and measured
        #at 2.3x SLOWER: with the anchor embedding as a lateral outer column
        #postgres re-detoasts the ~3kb vector for every one of the 61k
        #distance evaluations, while a bound parameter gets detoasted once.
        #
        #no l.id tiebreak on the ORDER BY: a second sort key pushes the
        #planner off the index and back onto the full scan. the 400 cut
        #stays deterministic anyway, walking an unchanged graph returns the
        #same rows in the same order, and only the ingest changes the graph
        with pool.connection() as c:
            return c.execute("""
                SELECT l.oracle_id, l.line_text, l.face, 1 - (l.embedding <=> %s) AS sim, """ + pcol + """ AS price, c.edhrec_rank
                FROM lines l JOIN cards c ON c.oracle_id = l.oracle_id
                WHERE l.oracle_id <> %s AND NOT l.whole""" + where + """
                ORDER BY l.embedding <=> %s
                LIMIT 400
            """, [ql["embedding"], oracle_id] + fparams + [ql["embedding"]]).fetchall()

    #multi-line cards pay for their scans side by side instead of one after
    #the other, each hunt on its own pooled connection. the main thread
    #holds NO connection while they run (holding one while workers wait on
    #the pool is how a pool deadlocks), and 3 workers keeps a connection
    #free for /suggest even when two searches land at once. map preserves
    #line order, so results merge exactly as the sequential loop did
    if len(qlines) > 1:
        with ThreadPoolExecutor(max_workers=min(3, len(qlines))) as ex:
            per_line = list(ex.map(hunt, qlines))
    else:
        per_line = [hunt(ql) for ql in qlines]

    for ql, matches in zip(qlines, per_line):
        w = line_weight(ql["count"])
        for m in matches:
            if m["oracle_id"] not in pairs_by_card:
                pairs_by_card[m["oracle_id"]] = []
            pairs_by_card[m["oracle_id"]].append((m["sim"] * w, m["sim"], ql["line_text"], m["line_text"], m["face"]))
            prices[m["oracle_id"]] = m["price"]
            ranks[m["oracle_id"]] = m["edhrec_rank"]

    with pool.connection() as conn:
        #sort each card's pairs so pairs[0] is its best one, then rank the
        #cards by that best pair
        ranked = []
        for oid, pairs in pairs_by_card.items():
            pairs.sort(reverse=True)
            ranked.append((oid, pairs))
        ranked.sort(key=lambda x: x[1][0][0], reverse=True)

        #the concepts side of the slider. one promise keeps it honest: the
        #badge is (1-w) * mech + w * concept, and the min-match line cuts on
        #that SAME number, so nothing under the cutoff ever shows above the
        #fold no matter which axis it leaned on
        concept_raw = {}
        if blend > 0:
            #cards the lines never found, injected as candidates when their
            #concept score alone is worth considering at the current cutoff,
            #through the same filters as everything else
            rows = conn.execute("""
                WITH anchor AS (
                    SELECT ct.tag, t.idf FROM card_tags ct
                    JOIN tags t ON t.tag = ct.tag
                    WHERE ct.oracle_id = %s
                )
                SELECT ct.oracle_id, """ + pcol + """ AS price, c.edhrec_rank,
                       sum(a.idf * a.idf) / (na.norm * nc.norm) AS raw
                FROM card_tags ct
                JOIN anchor a ON a.tag = ct.tag
                JOIN cards c ON c.oracle_id = ct.oracle_id
                JOIN card_tag_norms nc ON nc.oracle_id = ct.oracle_id
                JOIN card_tag_norms na ON na.oracle_id = %s
                WHERE ct.oracle_id <> %s""" + where + """
                GROUP BY ct.oracle_id, """ + pcol + """, c.edhrec_rank, na.norm, nc.norm
                HAVING sum(a.idf * a.idf) / (na.norm * nc.norm) >= %s
                ORDER BY raw DESC
                LIMIT 300
            """, [oracle_id, oracle_id, oracle_id] + fparams + [concept_raw_gate(min_pct)]).fetchall()
            for r in rows:
                concept_raw[r["oracle_id"]] = r["raw"]
                prices.setdefault(r["oracle_id"], r["price"])
                ranks.setdefault(r["oracle_id"], r["edhrec_rank"])

            #every mechanical candidate needs its concept score too, the
            #blend weighs both axes for everyone
            have = {oid for oid, pairs in ranked}
            ids = [oid for oid in have if oid not in concept_raw]
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

            #pure concept finds carry no line pairs, their badge is w * concept
            for oid in concept_raw:
                if oid not in have:
                    ranked.append((oid, []))

            def blended(entry):
                oid, pairs = entry
                mech = mech_display(pairs[0][1]) if pairs else 0
                return (1 - blend) * mech + blend * concept_display(concept_raw.get(oid, 0.0))
            ranked.sort(key=blended, reverse=True)
            gate_score = blended
        else:
            def gate_score(entry):
                return mech_display(entry[1][0][1])

        #split at the minimum match line. the number that decides which side
        #a card lands on is exactly the number its badge will show (rounded
        #the same way, so a 79.6 that badges as 80 passes)
        strong = []
        weak_tier = []
        for entry in ranked:
            if round(gate_score(entry)) >= min_pct:
                strong.append(entry)
            else:
                weak_tier.append(entry)
        weak_count = len(weak_tier)
        if weak:
            wanted = weak_tier
        else:
            wanted = strong

        #the alternate sorts happen after the filters, the ranking and the
        #tier split, so the percent keeps meaning what it always meant and
        #weak cards can never leapfrog strong ones. cards with no price sink
        #to the bottom of the price sorts, and ties fall back to the badge
        #score, which also covers concept-found cards with no line pairs
        if sort in ("cheap", "pricey"):
            priced = []
            unpriced = []
            for entry in wanted:
                if prices[entry[0]] is None:
                    unpriced.append(entry)
                else:
                    priced.append(entry)
            if sort == "cheap":
                priced.sort(key=lambda x: (float(prices[x[0]]), -gate_score(x)))
            else:
                priced.sort(key=lambda x: (-float(prices[x[0]]), -gate_score(x)))
            wanted = priced + unpriced
        elif sort in ("played", "obscure"):
            #edhrec rank, 1 = the format's most played card. no rank reads
            #as nobody plays it, so unranked cards sink on the played sort
            #and float on the obscure one
            worst = 10 ** 9
            flip = 1 if sort == "played" else -1
            wanted = sorted(wanted, key=lambda x: (flip * (ranks.get(x[0]) or worst), -gate_score(x)))

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
            score, sim, our_line, their_line, their_face = pairs[0]
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
            their_face = 0
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
        price = price_label(c, currency)
        #a match that lives on the back face shows that side first, so the
        #line printed under the card is on the picture the user is looking
        #at (the ulvenwald lesson). the front face keeps the flip button
        image = c["image"]
        image_back = c["image_back"] or ""
        matched_back = their_face == 1 and bool(image_back)
        if matched_back:
            image, image_back = image_back, image
        results.append({
            "oracle_id": str(oid),  #the report flag needs to say which card it's flagging
            "name": c["name"],
            "mana_cost": c["mana_cost"],
            "type_line": c["type_line"],
            "image": image,
            "image_back": image_back,
            "matched_back": matched_back,
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
            "rank": rank_label(c["edhrec_rank"]),
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
    blend = read_blend()
    min_pct = read_min(blend)
    #the note with the keep-my-min button only shows when the relaxed
    #default actually kicked in, never over a min the user chose
    min_default = default_min(blend)
    min_auto = request.args.get("min") is None and min_default == 70
    #the mirror image: a chosen min wins over the default from here on, so
    #when it is not the default anyway, offer the way back to it
    min_override = request.args.get("min") is not None and min_pct != min_default
    sort = read_sort()
    card_lines, picked = build_lines(card, read_picked())

    results, has_more, weak_count = find_similar(card["oracle_id"], picked, filters, min_pct, sort,
                                                 blend=BLEND_WEIGHTS[blend], currency=filters["cur"])
    resp = make_response(render_template("search.html", query=query, card=card, card_lines=card_lines,
                                         picked_count=len(picked), results=results, has_more=has_more,
                                         weak_count=weak_count, min_pct=min_pct, min_auto=min_auto,
                                         min_override=min_override, min_default=min_default,
                                         blend=blend, cur=filters["cur"], types=CARD_TYPES))
    #an explicit slider position becomes the remembered one. moving it back
    #to rules text remembers that too, so there is no stuck state
    if request.args.get("blend") is not None:
        resp.set_cookie("blend", str(blend), max_age=60 * 60 * 24 * 365, samesite="Lax")
    return resp


@app.route("/guide")
def guide():
    #the how-it-works page. the demo card is fetched live so its picture
    #stays current, and the page just skips the demo if it ever vanishes.
    #a transform card on purpose, so the legend can point at the transform
    #button the card-frame overlay puts on two-faced cards
    demo = find_card("Delver of Secrets")
    return render_template("guide.html", demo=demo)


@app.route("/unique")
def unique():
    return render_template("unique.html", types=CARD_TYPES, blend=read_blend(), cur=read_currency())


def card_json(c, currency):
    #one dealt (or revisited) card the way the /unique frontend wants it.
    #layout and image_back are what let the page offer the rotate and
    #turn-over buttons
    price = price_label(c, currency)
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
        "rank": rank_label(c["edhrec_rank"]),
        "percent": int(round((c.get("blended_u") if c.get("blended_u") is not None else (c["uniqueness"] or 0)) * 100)),
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
    body = request.get_json(silent=True) or {}
    seen = []
    #the browser caps its list at 2000, the [-4000:] is the server not
    #taking its word for it: newest entries win, a hand-rolled megalist
    #can't make the query chew through millions of uuids
    for s in body.get("seen", [])[-4000:]:
        #only real uuids get through to the query, anything else in
        #localStorage was not put there by us
        try:
            seen.append(str(uuid.UUID(str(s))))
        except ValueError:
            pass

    where, fparams = filter_sql(filters)
    #no uniqueness bar anymore: the dealer takes the 100 most unique unseen
    #cards that fit the filters and deals one at random. the window slides
    #down as the trail grows, so there is always a next card until the
    #filters truly run dry, and nobody has to learn what a uniqueness
    #number means. the slider decides which KIND of unique: rules-text
    #isolation, tag-space isolation, or a mix. cards with no searchable
    #lines stay excluded, untagged cards count as 0 on the concept side
    w = BLEND_WEIGHTS[read_blend()]
    blended = "((1 - %s) * c.uniqueness + %s * coalesce(c.concept_uniqueness, 0))"
    cond = """
        FROM cards c
        WHERE c.uniqueness IS NOT NULL
          AND NOT (c.oracle_id = ANY(%s::uuid[]))""" + where
    params = [seen] + fparams
    with pool.connection() as conn:
        remaining = conn.execute("SELECT count(*) AS n" + cond, params).fetchone()["n"]
        rows = conn.execute("SELECT * FROM (SELECT " + CARD_FIELDS + ", uniqueness, unique_line, "
                            + blended + " AS blended_u" + cond + """
                            ORDER BY blended_u DESC LIMIT 100) top_window
                            ORDER BY random() LIMIT %s""",
                            [w, w] + params + [UNIQUE_PAGE]).fetchall()

    cards = []
    cur = read_currency()
    for c in rows:
        cards.append(card_json(c, cur))
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
        #the trail arrows show the same blended number a fresh deal would,
        #using the remembered slider position
        w = BLEND_WEIGHTS[read_blend()]
        c = conn.execute("SELECT " + CARD_FIELDS + """, uniqueness, unique_line,
                            ((1 - %s) * uniqueness + %s * coalesce(concept_uniqueness, 0)) AS blended_u
                          FROM cards WHERE oracle_id = %s""",
                         (w, w, oid)).fetchone()
    if c is None:
        return {"card": None}
    return {"card": card_json(c, read_currency())}


#the load more button on the results page calls this and gets json back. it
#receives the page's whole query string, so filters and picked lines apply
@app.route("/more")
def more():
    query = request.args.get("q", "")
    #fail-soft like every other url reader, a doctored offset shouldn't 500
    try:
        offset = max(0, int(request.args.get("offset", 0)))
    except ValueError:
        offset = 0
    card = find_card(query)
    if card is None:
        return {"results": [], "has_more": False, "weak_count": 0}
    card_lines, picked = build_lines(card, read_picked())
    weak = request.args.get("weak") == "1"
    blend = read_blend()
    filters = read_filters()
    results, has_more, weak_count = find_similar(card["oracle_id"], picked, filters, read_min(blend), read_sort(), offset,
                                                 weak=weak, blend=BLEND_WEIGHTS[blend], currency=filters["cur"])
    return {"results": results, "has_more": has_more, "weak_count": weak_count}


#---- user feedback: "a card is missing" / "this card shouldn't be here" ----

def client_ip():
    #railway's proxy APPENDS the address it saw to X-Forwarded-For, so the
    #last entry is its word and everything left of it is client supplied.
    #reading the first entry would let anyone dodge the report rate limit by
    #sending a made-up header. one proxy deep is a railway fact: putting a
    #cdn in front of the site would add an entry and this needs to move one
    #step left
    fwd = request.headers.get("X-Forwarded-For", "")
    if fwd:
        return fwd.split(",")[-1].strip()
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
        JOIN lines b ON b.oracle_id = %s AND NOT b.whole
        LEFT JOIN line_stats s ON s.line_text = a.line_text
        WHERE a.oracle_id = %s AND NOT a.whole
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
    #the same currency the bounds filtered on, so the number quoted back is
    #the one the user was comparing against
    pkey = "price_usd" if filters.get("cur", "usd") == "usd" else "price_eur"
    sign = "$" if pkey == "price_usd" else "€"
    if (filters["pmin"] is not None or filters["pmax"] is not None) and card[pkey] is None:
        reasons.append("it has no listed price, and any price filter hides unpriced cards")
    elif filters["pmin"] is not None and float(card[pkey]) < filters["pmin"]:
        reasons.append("its price (" + sign + str(card[pkey]) + ") is under your minimum")
    elif filters["pmax"] is not None and float(card[pkey]) > filters["pmax"]:
        reasons.append("its price (" + sign + str(card[pkey]) + ") is over your maximum")
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
    blend = read_blend()
    min_pct = read_min(blend)
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
            full = conn.execute("""SELECT color_identity, price_usd, price_eur, cmc, type_line, game_changer, legal_commander, oracle_text
                                   FROM cards WHERE oracle_id = %s""", (expected["oracle_id"],)).fetchone()
            reasons = filter_reasons(full, filters)
            #the filter box is one compiled expression, so the honest check
            #is asking the database whether this card survives it
            if filters.get("fq_sql") and not conn.execute(
                    "SELECT 1 FROM cards c WHERE c.oracle_id = %s AND (" + filters["fq_sql"] + ")",
                    [expected["oracle_id"]] + filters["fq_params"]).fetchone():
                reasons.append("your filter query hides it")
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
    out += "    **Anchor:** " + r["anchor_name"] + " - " + q(anchor_lines) + "\n"
    if r["kind"] == "misplaced":
        out += "    **NOT:** " + (r["got_name"] or "?") + " - " + q(line_texts.get(r["got_id"], [])) + "\n"
        out += "    *user report " + day + "; the flagged card showed at " + str(r["got_pct"]) + "% mech" + mode + "; reason: " + r["reason"] + "*\n"
    else:
        out += "    **Match:** " + (r["expected_name"] or "?") + " - " + q(line_texts.get(r["expected_id"], [])) + "\n"
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
            for l in conn.execute("SELECT oracle_id, line_text FROM lines WHERE oracle_id = ANY(%s) AND NOT whole", (list(ids),)):
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
#then trigram matches to catch typos. one query, tiered like find_card: the
#substring tiers read alphabetically, the fuzzy tier closest-first (its
#alphabetical CASE key is NULL, which sorts after every real name). this is
#the hottest route on the site, it fires on every pause in typing
@app.route("/suggest")
def suggest():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return {"names": []}
    p, s = q + "%", "%" + q + "%"
    names = []
    with pool.connection() as conn:
        for row in conn.execute("""
            SELECT name,
                   CASE WHEN name ILIKE %s THEN 0
                        WHEN name ILIKE %s THEN 1
                        ELSE 2 END AS tier
            FROM cards
            WHERE name ILIKE %s OR name ILIKE %s OR name %% %s
            ORDER BY tier, CASE WHEN name ILIKE %s THEN name END, name <-> %s, name
            LIMIT 8
        """, (p, s, p, s, q, s, q)):
            if row["name"] not in names:  #the odd duplicated name collapses to one entry
                names.append(row["name"])
    return {"names": names}


#---- crawler plumbing: robots.txt and the sitemap ----

#every card's search page is a landing page, but crawlers can only find
#them by walking result pages link by link. the sitemap hands over the
#whole list of canonical card urls in one file. the names are cached for a
#day (they change on the ingest's schedule, not the request's), the xml is
#rebuilt per request because it embeds whichever host the request came in on
_sitemap_names = {"names": [], "made": 0.0}


@app.route("/sitemap.xml")
def sitemap():
    now = time.time()
    if not _sitemap_names["names"] or now - _sitemap_names["made"] > 60 * 60 * 24:
        with pool.connection() as conn:
            _sitemap_names["names"] = [r["name"] for r in conn.execute("SELECT name FROM cards ORDER BY name")]
        _sitemap_names["made"] = now
    root = request.url_root
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for page in ("", "unique", "guide"):
        out.append("<url><loc>" + root + page + "</loc></url>")
    for name in _sitemap_names["names"]:
        #quote() with its defaults mirrors the urlencode filter building the
        #canonicals in search.html, so these are the urls the pages declare.
        #it also percent-encodes every xml-special character, & included, so
        #the raw name never needs xml escaping
        out.append("<url><loc>" + root + "search?q=" + quote(name) + "</loc></url>")
    out.append("</urlset>")
    #text/xml instead of application/xml so flask-compress gzips it. the
    #protocol caps one sitemap at 50k urls, the card pool sits well under
    return Response("\n".join(out), mimetype="text/xml")


@app.route("/robots.txt")
def robots():
    #the disallows are the json endpoints the pages fetch, nothing a search
    #result should point at. every human page stays open, and the sitemap
    #line lets crawlers find the card list without a console submission
    return Response("\n".join([
        "User-agent: *",
        "Disallow: /suggest",
        "Disallow: /more",
        "Disallow: /unique/",
        "Sitemap: " + request.url_root + "sitemap.xml",
    ]) + "\n", mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True)
