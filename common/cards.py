#card handling helpers shared between the web app and the update pipeline.
#all of this is lifted straight out of the old build_index.py so the results
#stay identical to the file based version

import re

from common.prefix_words import PREFIX_WORDS

#scryfall's docs say to send a real user agent with api requests
HEADERS = {"User-Agent": "Delvefall/1.0 (personal project)", "Accept": "application/json"}

#layouts that arent actual playable cards
SKIP_LAYOUTS = ["token", "double_faced_token", "emblem", "art_series", "planar", "scheme", "vanguard"]


def get_text(card):
    #double faced cards keep their text on the faces instead of the card itself
    if card.get("oracle_text"):
        return card["oracle_text"]
    if "card_faces" in card:
        parts = []
        for face in card["card_faces"]:
            if face.get("oracle_text"):
                parts.append(face["oracle_text"])
        return "\n".join(parts)
    return ""


def get_image(card):
    if "image_uris" in card:
        return card["image_uris"].get("normal", "")
    if "card_faces" in card and "image_uris" in card["card_faces"][0]:
        return card["card_faces"][0]["image_uris"].get("normal", "")
    return ""


def get_back_image(card):
    #the back face's picture, for the turn-the-card-over button. only double
    #faced layouts carry per-face image_uris (split and adventure cards have
    #two faces too, but those share one picture and land in get_image above)
    faces = card.get("card_faces")
    if faces and len(faces) > 1 and "image_uris" in faces[1]:
        return faces[1]["image_uris"].get("normal", "")
    return ""


#mechanics whose reminder text IS the rule. "Overload {6}{U}" says nothing on
#its own, so stripping the parens stored Cyclonic Rift as a plain one-target
#bounce spell and it matched Perilous Voyage at 91%, missing the one-sided
#board wipe that makes the card worth $30.
#
#evergreen abilities are deliberately absent. 2442 cards print a bare "Flying"
#against 75 that spell the reminder out, so keeping those 75 would orphan them
#from the other 2442 - the exact opposite of the fix. every keyword here was
#measured to be printed WITH its reminder at least ~88% of the time, so the
#whole population moves together instead of splitting in half. equip (242 with
#against 332 without) and menace and trample fail that test and stay stripped.
#check the ratio before adding to this list
REMINDER_KEYWORDS = {
    "overload", "cascade", "storm", "cycling", "flashback", "morph", "disguise",
    "madness", "convoke", "delve", "buyback", "entwine", "replicate", "embalm",
    "eternalize", "unearth", "disturb", "blitz", "bargain", "craft", "mutate",
    "foretell", "bestow", "improvise", "emerge", "evoke", "dash", "spectacle",
    "surge", "escalate", "splice", "rebound", "conspire", "retrace", "miracle",
    "ninjutsu", "prowl", "transmute", "scavenge", "encore", "outlast",
}

#one keyword name plus any mana symbols, eg "Cycling {2}" or "Flying"
_BARE_KEYWORD = re.compile(r"[A-Za-z][A-Za-z'’ -]*(?:\s*\{[^}]*\})*")


def reminder_is_the_rule(stripped):
    #true when removing the parens left nothing but keyword names and mana
    #symbols ("Cycling {2}", "Flying, double strike") AND the leading keyword
    #is one whose reminder text carries the actual rule. anything with a real
    #sentence in it kept its meaning and doesnt need the reminder back
    text = stripped.strip().rstrip(".")
    if not text:
        return False
    for part in text.split(","):
        part = part.strip()
        if part and not _BARE_KEYWORD.fullmatch(part):
            return False
    first = re.split(r"[^A-Za-z'’-]", text, maxsplit=1)[0].lower()
    return first in REMINDER_KEYWORDS


def clean_line(line, card_name):
    #reminder text (the stuff in parens) is just for humans, the model doesnt
    #need it - except where the parens hold the whole rule, see above
    stripped = re.sub(r"\(.*?\)", "", line)
    if reminder_is_the_rule(stripped):
        line = line.replace("(", "").replace(")", "")
    else:
        line = stripped
    #flavour prefixes must not beat meaning (testing_list CA): die-roll table
    #rows ("1—9 |"), saga chapter markers ("I, II —") and ability/flavor
    #words ("Landfall —", "Siege Monster —") say when or in what style, not
    #what happens, so they go. the word list is scryfall's own catalogs, so
    #keywords that genuinely use the dash (Boast, Companion) stay whole
    line = re.sub(r"^\d+(?:—\d+)?\s*\|\s*", "", line)
    line = re.sub(r"^[IVX]+(?:, [IVX]+)*\s+—\s+", "", line)
    m = re.match(r"^([^—•|]{1,40}?)\s+—\s+(?=\S)", line)
    if m and m.group(1) in PREFIX_WORDS:
        line = line[m.end():]
    #cards refer to themselves by name, which would make the model think names
    #matter. swap it for something generic. legendary cards also get shortened
    #to their first name in the middle of the text ("Jacob, the Great" -> "Jacob")
    #so handle that too
    line = line.replace(card_name, "this card")
    if "," in card_name:
        line = line.replace(card_name.split(",")[0], "this card")
    return line.strip()


def keep_card(card):
    #same filters the old builder used, plus a check that scryfall actually
    #gave us an oracle_id since thats the primary key in the database now
    if not card.get("oracle_id"):
        return False
    if card.get("set_type") in ("funny", "memorabilia"):
        return False  #skip the joke sets
    if card.get("layout") in SKIP_LAYOUTS:
        return False
    if card.get("digital") and card.get("legalities", {}).get("vintage", "not_legal") == "not_legal":
        #arena/mtgo only cards (alchemy rebalances etc), never printed in paper.
        #the vintage check matters: scryfall sometimes picks a digital printing
        #to represent a real paper card (ancestral recall arrives as vintage
        #masters, an mtgo set), and every real paper card is at least restricted
        #or banned in vintage, so those stay. true digital-only cards are
        #not_legal there and still get dropped
        return False
    if not get_text(card).strip():
        return False  #vanilla creatures, basic lands etc, nothing to compare
    return True


def split_lines(card):
    #each line of rules text is basically one ability, so embed every line
    #separately instead of whole cards. that way one matching ability is
    #enough. every line remembers which face printed it (0 front, 1 back),
    #so a match that lives on the back face can show that side of the card
    if card.get("oracle_text"):
        chunks = [(card["oracle_text"], 0)]
    else:
        chunks = [(f.get("oracle_text", ""), i) for i, f in enumerate(card.get("card_faces", []))]
    out = []
    for text, face in chunks:
        for line in text.split("\n"):
            cleaned = clean_line(line, card["name"])
            if len(cleaned) < 3:
                continue
            out.append((cleaned, min(face, 1)))
    return out
