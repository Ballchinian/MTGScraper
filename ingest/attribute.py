#works out which tags each LINE of a card is about, so the search page can
#narrow the concept axis to the ability you picked instead of always scoring
#the whole card's tag vector.
#
#the problem this solves: tagger tags CARDS. a card tagged donate-token,
#gives-pp-counters-to-all and evasion offers no way to know that the first
#belongs to its token mode and the last to its "Flying, double strike" line,
#so picking one line used to change the rules-text axis and leave the concept
#axis searching all of them at once.
#
#the inference is corpus-shaped rather than semantic. for a line, pull its
#nearest neighbour lines from every OTHER card, then ask of each of its card's
#tags: what share of those neighbour cards carry this tag, against the share
#the whole game carries it? that ratio is the lift, and a high one means this
#line is why the card got the tag. it needs no model and no understanding:
#"Overload {6}{U}" carries no meaning at all, but its neighbours are other
#overload cards, and those are tagged sweeper-one-sided, so the tag lands on
#the right line anyway.
#
#run it from the repo root, after the card and tag ingests:
#    python -m ingest.attribute
#with DATABASE_URL set. needs numpy and psycopg, no torch and no model, since
#every embedding it reads is already in the database.

import os
import re
import sys
import math
import collections

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

#how many neighbour lines vote. 200 is wide enough that a common line still
#gathers a varied neighbourhood and narrow enough that a rare one doesn't
#reach past its real family into noise
NEIGHBOURS = 200

#a tag has to appear in a line's neighbourhood at least this many times more
#often than in the game at large before that line is credited with it. 1.5 is
#deliberately low: the ratio decides which line OWNS a tag, and the floor only
#exists to reject lines whose neighbourhood is indifferent to it. evergreen
#tags sit near the bottom of this range on purpose ("Flying, double strike"
#lifts evasion 2.4x, which is weak but still the right line for it)
FLOOR = 1.5

#below this a line has no claim on a tag at all. 1.0 is the neutral point of
#a lift ratio (the neighbourhood carries the tag exactly as often as the game
#does), so anything at or under it is evidence of nothing and the tag is left
#off every line rather than parked on the least-bad one
NOISE = 1.15

#how much the second, completely different signal counts for. tagger writes a
#plain english description of most tags ("At the end of turn, if a condition
#isn't met, a consequence is given" for catch-22) and the lines are plain
#english too, so word overlap between the two is evidence the neighbourhood
#vote knows nothing about. the two fail in different places, which is the
#whole point: the vote is strong on tags with many examples and blind on
#abstract ones, the words are strong on tags whose description names what the
#line does and blind on jargon like cantrip. measured on the hand-labelled
#cards, 0.5 took precision from 78% to 90% at the same recall. above ~1.0 the
#words start outvoting the corpus and precision falls again
LEX_WEIGHT = 0.5

#once a tag's best line is known, any other line within this fraction of that
#best also gets it. modal cards are why: each mode line lifts "modal" hard,
#and crediting only the single strongest would make picking any other mode
#silently drop the tag.
#
#tuned for precision rather than recall: a tag set aside by mistake can be
#clicked back on (the page's yestags), so a miss costs one click, while a
#wrong tag quietly drags the whole search sideways. measured against a
#hand-labelled Shadrix Silverquill, 0.4 gives 88% precision / 82% recall and
#0.6 gives 93% / 76%. that is ONE card, so re-measure on a second labelled
#card before trusting the third digit
RATIO = 0.6


def _words(text):
    #crude but deterministic: lowercase, letters only, drop the very short
    #ones, and chop the common suffixes so "creatures"/"creature" and
    #"gains"/"gain" line up. a real stemmer would be better and would also be
    #a new dependency for a signal this small
    out = []
    for w in re.findall(r"[a-z]+", (text or "").lower()):
        if len(w) < 3:
            continue
        for suffix in ("ing", "es", "ed", "s"):
            if len(w) > 4 and w.endswith(suffix):
                w = w[:-len(suffix)]
                break
        out.append(w)
    return out


def build_lexical(conn):
    #word idf over the line corpus, so "creature" and "target" count for
    #almost nothing and "planeswalker" or "haste" carry the match. returns a
    #scorer for one line against one tag
    lines_seen = 0
    doc_freq = collections.Counter()
    for (text,) in conn.execute("SELECT line_text FROM lines WHERE NOT whole"):
        lines_seen += 1
        for w in set(_words(text)):
            doc_freq[w] += 1
    idf = {w: math.log(lines_seen / (1 + n)) for w, n in doc_freq.items()}
    unseen = math.log(max(lines_seen, 2))

    #a tag's text is its own slug plus whatever description tagger wrote.
    #only ~42% of tags have a description, so this signal is silent for the
    #rest and the neighbourhood vote carries them alone
    tag_words = {}
    for tag, description in conn.execute("SELECT tag, description FROM tags"):
        tag_words[tag] = set(_words(tag.replace("-", " ")) + _words(description))

    def score(line_text, tag):
        wanted = tag_words.get(tag)
        if not wanted:
            return 0.0
        shared = wanted & set(_words(line_text))
        if not shared:
            return 0.0
        #divided by the tag's own weight so a wordy description cannot
        #outscore a terse one just by having more chances to match
        total = sum(idf.get(w, unseen) for w in wanted)
        return sum(idf.get(w, unseen) for w in shared) / math.sqrt(total + 1)

    return score


def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("set DATABASE_URL first (the postgres connection string)")
        sys.exit(1)

    conn = psycopg.connect(db_url)
    register_vector(conn)  #without this the embeddings arrive as strings
    schema_path = os.path.join(os.path.dirname(__file__), "..", "common", "schema.sql")
    with open(schema_path, encoding="utf-8") as f:
        conn.execute(f.read())
    conn.commit()

    total_cards = conn.execute("SELECT count(*) FROM cards").fetchone()[0]
    if not total_cards:
        print("no cards yet, nothing to attribute")
        return

    #every tag a card carries, rolled up, is what the neighbours vote WITH.
    #only the typed ones get attributed though: the inherited ancestors follow
    #from the tree at query time, the same way they do for a whole card
    print("reading tags...")
    all_tags = {}
    typed_tags = {}
    for oid, tag, inherited in conn.execute("SELECT oracle_id, tag, inherited FROM card_tags"):
        oid = str(oid)
        all_tags.setdefault(oid, set()).add(tag)
        if not inherited:
            typed_tags.setdefault(oid, set()).add(tag)
    base_rate = {}
    for tag, count in conn.execute("SELECT tag, card_count FROM tags"):
        base_rate[tag] = max(count, 1) / total_cards
    print("  " + str(len(typed_tags)) + " cards carry at least one typed tag")

    #whole-card rows stay out, exactly like every other line-shaped pass
    print("reading line embeddings...")
    ids = []
    owners = []
    line_text = []
    vecs = []
    for lid, oid, text, vec in conn.execute(
            "SELECT id, oracle_id, line_text, embedding FROM lines WHERE NOT whole ORDER BY id"):
        ids.append(lid)
        owners.append(str(oid))
        line_text.append(text)
        vecs.append(vec.to_numpy())  #pgvector hands back its own Vector class
    if not ids:
        print("no lines yet, nothing to attribute")
        return
    emb = np.asarray(vecs, dtype=np.float32)
    del vecs
    print("  pulled " + str(len(ids)) + " embeddings")

    rows_of_card = {}
    for i, oid in enumerate(owners):
        rows_of_card.setdefault(oid, []).append(i)

    #same blocked multiply as the uniqueness pass: the embeddings are
    #normalized so cosine is a plain dot product, and blocking keeps the
    #similarity matrix around 100mb instead of the 13gb the whole thing would
    #need. argpartition beats a sort here, nothing cares about the order
    #within a neighbourhood, only who is in it
    print("finding neighbourhoods...")
    k = min(NEIGHBOURS, len(ids) - 1)
    neighbours = np.zeros((len(ids), k), dtype=np.int32)
    block = 512
    for start in range(0, len(ids), block):
        sims = emb[start:start + block] @ emb.T
        for r in range(sims.shape[0]):
            sims[r, rows_of_card[owners[start + r]]] = -2.0  #a card never votes on itself
        neighbours[start:start + block] = np.argpartition(sims, -k, axis=1)[:, -k:]
        if start % (block * 20) == 0:
            print("  " + str(start) + "/" + str(len(ids)))
    del emb

    #how many lines each card has, for the first pass's damping below
    lines_on_card = {oid: len(idxs) for oid, idxs in rows_of_card.items()}

    def assign(lift_of, lexical=None):
        #per card and per tag, which of its lines earned it. the best line
        #sets the bar and everything within RATIO of it shares the credit.
        #
        #a tag no line shows any evidence for is attributed to NOTHING, and
        #that is deliberate. it used to ride every line instead, on the theory
        #that tags like invitational-card describe the card rather than an
        #ability - but that is exactly why they should be absent: they are not
        #about any ability, so picking an ability should drop them. whole-card
        #searches never read this table, so nothing is lost there, and Omnath's
        #unique-mana-cost stops turning up under "when this card enters, draw a
        #card". attaching them everywhere was the single largest source of
        #false positives on the hand-labelled cards.
        #
        #when a lexical scorer is supplied the two signals are each normalized
        #against their own best line before being added, because a lift ratio
        #and a word-overlap score share no scale. normalizing per tag also
        #means only the SHAPE across this card's lines matters, which is the
        #only question being asked
        out = {}
        for oid, line_idxs in rows_of_card.items():
            for tag in typed_tags.get(oid, ()):
                lifts = [(i, lift_of.get((i, tag), 0.0)) for i in line_idxs]
                best = max(l for _, l in lifts)
                if lexical is not None:
                    lex = [lexical(line_text[i], tag) for i in line_idxs]
                    best_lex = max(lex)
                    if best < NOISE and best_lex <= 0:
                        continue  #neither signal has anything to say
                    if best_lex > 0:
                        #blend, then re-express on the lift scale so the bar
                        #and floor logic below reads the same as it always did
                        scale = best if best > 0 else 1.0
                        blended = [((l / scale) if best > 0 else 0.0)
                                   + LEX_WEIGHT * (x / best_lex)
                                   for (_, l), x in zip(lifts, lex)]
                        top = max(blended)
                        lifts = [(i, (b / top) * max(best, NOISE))
                                 for (i, _), b in zip(lifts, blended)]
                        best = max(l for _, l in lifts)
                #a lift of 1.0 means the neighbourhood carries the tag at
                #exactly the rate the whole game does, which is no evidence
                #whatsoever. Omnath's unique-mana-cost sat at 1.0x on "when
                #this card enters, draw a card" and got attributed anyway,
                #because the only bar was "above zero"
                if best < NOISE:
                    continue
                #near-best only when the signal is weak, since RATIO of a
                #small number would wave nearly every line through
                bar = max(best * RATIO, FLOOR) if best >= FLOOR else best * 0.9
                for i, l in lifts:
                    if l >= bar:
                        out[(i, tag)] = l
        return out

    #pass one: neighbours vote with their whole CARD's tags, because
    #card-level tags are all there is to start from. that is also its flaw - a
    #neighbour card with five lines donates all five lines' worth of tags to
    #whichever one line matched - so each neighbour's vote is damped by how
    #many lines it has. a one-line card knows exactly which line earned its
    #tags and speaks at full volume
    print("scoring, pass one (cards vote)...")
    lift_of = {}
    for i in range(len(ids)):
        mine = typed_tags.get(owners[i])
        if not mine:
            continue
        votes = {}
        for j in neighbours[i]:
            cid = owners[j]
            if cid != owners[i]:
                votes[cid] = 1.0 / lines_on_card.get(cid, 1)
        if not votes:
            continue
        total_vote = sum(votes.values())
        for tag in mine:
            hit = 0.0
            for cid, w in votes.items():
                if tag in all_tags.get(cid, ()):
                    hit += w
            lift_of[(i, tag)] = (hit / total_vote) / base_rate.get(tag, 1.0)
    first = assign(lift_of)

    #pass two: now that every line has a provisional guess, neighbours vote
    #with their own LINE's tags instead of their card's, which is the thing
    #pass one could not do. measured against a hand-labelled card this lifts
    #precision from 60% to 88% at the same neighbourhood, because a line that
    #merely sits on a card with an unrelated ability stops donating it
    print("scoring, pass two (lines vote)...")
    line_tags_now = {}
    for (i, tag) in first:
        line_tags_now.setdefault(i, set()).add(tag)
    lift2 = {}
    for i in range(len(ids)):
        mine = typed_tags.get(owners[i])
        if not mine:
            continue
        nb = [j for j in neighbours[i] if owners[j] != owners[i]]
        if not nb:
            continue
        for tag in mine:
            hits = 0
            for j in nb:
                if tag in line_tags_now.get(j, ()):
                    hits += 1
            lift2[(i, tag)] = (hits / len(nb)) / base_rate.get(tag, 1.0)

    #pass two sharpens, it does not get to erase. a rare tag can be real on
    #one line and still have no neighbour line carrying it yet, and pass two
    #reads zero for that - Omnath's sweeper-one-sided is the case that caught
    #it. so pass two's answer wins wherever it found anything at all for a
    #tag, and pass one's stands where it found nothing
    #the word signal only joins at the end. it is an independent opinion about
    #which line a tag belongs to, not something the corpus votes should be
    #bootstrapped from - folding it into pass one would let it contaminate the
    #neighbourhood evidence that pass two then reads back
    print("reading tag descriptions...")
    lexical = build_lexical(conn)

    print("assigning...")
    second = assign(lift2, lexical)
    seen_in_second = {(owners[i], tag) for i, tag in second}
    final = dict(second)
    for (i, tag), l in assign(lift_of, lexical).items():
        if (owners[i], tag) not in seen_in_second:
            final[(i, tag)] = l

    rows = [(ids[i], tag, l, False) for (i, tag), l in final.items()]

    print("writing " + str(len(rows)) + " line-tag rows...")
    with conn.cursor() as cur:
        cur.execute("TRUNCATE line_tags")
        with cur.copy("COPY line_tags (line_id, tag, lift, card_level) FROM STDIN") as copy:
            for r in rows:
                copy.write_row(r)
    conn.commit()

    covered = conn.execute("SELECT count(DISTINCT line_id) FROM line_tags").fetchone()[0]
    card_level = conn.execute("SELECT count(*) FROM line_tags WHERE card_level").fetchone()[0]
    conn.close()
    print("done! " + str(covered) + "/" + str(len(ids)) + " lines carry tags, "
          + str(card_level) + " rows are card-level")


if __name__ == "__main__":
    main()
