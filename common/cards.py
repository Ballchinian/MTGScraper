#card handling helpers shared between the web app and the update pipeline.
#all of this is lifted straight out of the old build_index.py so the results
#stay identical to the file based version

import re

#scryfall's docs say to send a real user agent with api requests
HEADERS = {"User-Agent": "Cardalike/1.0 (personal project)", "Accept": "application/json"}

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


def clean_line(line, card_name):
    #reminder text (the stuff in parens) is just for humans, the model doesnt need it
    line = re.sub(r"\(.*?\)", "", line)
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
    if card.get("digital"):
        return False  #arena/mtgo only cards (alchemy rebalances etc), never printed in paper
    if not get_text(card).strip():
        return False  #vanilla creatures, basic lands etc, nothing to compare
    return True


def split_lines(card):
    #each line of rules text is basically one ability, so embed every line
    #separately instead of whole cards. that way one matching ability is enough
    out = []
    for line in get_text(card).split("\n"):
        cleaned = clean_line(line, card["name"])
        if len(cleaned) < 3:
            continue
        out.append(cleaned)
    return out
