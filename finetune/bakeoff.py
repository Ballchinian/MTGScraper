#the model bake-off: scores candidate embedding models against the hand built
#triplet list in triplets.md. a triplet passes when the anchor line lands
#closer to the should-match card than to the should-NOT card, where each
#candidate card's best matching line counts, same as the real engine.
#
#run from the repo root with any python that has sentence-transformers:
#    python finetune/bakeoff.py
#
#models download into the huggingface cache on first run (a few GB total).
#a model that fails to load (gated repo, missing deps) gets skipped with a
#note instead of killing the scoreboard. results also land in
#finetune/bakeoff_results.csv with the raw similarity numbers.

import os
import sys
import csv
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.cards import clean_line

#(label, huggingface id, prompt prepended to every line or None)
#embeddinggemma and qwen3 are trained to expect a task prompt, the others not
MODELS = [
    ("all-MiniLM-L6-v2 (current)", "sentence-transformers/all-MiniLM-L6-v2", None),
    ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", None),
    ("gte-modernbert-base", "Alibaba-NLP/gte-modernbert-base", None),
    ("EmbeddingGemma-300m", "google/embeddinggemma-300m", "task: sentence similarity | query: "),
    ("Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B", "Instruct: Retrieve semantically similar text.\nQuery: "),
    #the fine tune, trained by finetune/train.py on the generated data. it
    #learned with the sentence similarity prompt so it must always be used
    #with the same one
    ("mtg-tuned EmbeddingGemma", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mtg-tuned-embeddinggemma-300m"),
     "task: sentence similarity | query: "),
]

#raw oracle lines straight off scryfall (verified 2026-07-08 / 2026-07-12),
#cleaned below with the same clean_line the ingest pipeline uses. candidates
#carry every line of theirs that the engine would see, best one counts.
#format: (number, short label, (anchor card, anchor line),
#         (match card, [lines]), (not card, [lines]))
TRIPLETS = [
    (1, "loot vs rummage",
     ("Merfolk Looter", "{T}: Draw a card, then discard a card."),
     ("Careful Study", ["Draw two cards, then discard two cards."]),
     ("Rummaging Goblin", ["{T}, Discard a card: Draw a card."])),
    (2, "loot vs cost-discard",
     ("Frantic Search", "Draw two cards, then discard two cards. Untap up to three lands."),
     ("Merfolk Looter", ["{T}: Draw a card, then discard a card."]),
     ("Cathartic Reunion", ["As an additional cost to cast this spell, discard two cards.", "Draw three cards."])),
    (3, "tap vs untap",
     ("Pressure Point", "Tap target creature."),
     ("Frost Breath", ["Tap up to two target creatures. Those creatures don't untap during their controller's next untap step."]),
     ("Refocus", ["Untap target creature.", "Draw a card."])),
    (4, "dies vs enters",
     ("Blood Artist", "Whenever this creature or another creature dies, target player loses 1 life and you gain 1 life."),
     ("Zulaport Cutthroat", ["Whenever this creature or another creature you control dies, each opponent loses 1 life and you gain 1 life."]),
     ("Soul Warden", ["Whenever another creature enters, you gain 1 life."])),
    (5, "yours vs theirs",
     ("Soul Warden", "Whenever another creature enters, you gain 1 life."),
     ("Ajani's Welcome", ["Whenever a creature you control enters, you gain 1 life."]),
     ("Blood Seeker", ["Whenever a creature an opponent controls enters, you may have that player lose 1 life."])),
    (6, "hand vs battlefield",
     ("Raise Dead", "Return target creature card from your graveyard to your hand."),
     ("Gravedigger", ["When this creature enters, you may return target creature card from your graveyard to your hand."]),
     ("Zombify", ["Return target creature card from your graveyard to the battlefield."])),
    (7, "reanimate reworded",
     ("Zombify", "Return target creature card from your graveyard to the battlefield."),
     ("Reanimate", ["Put target creature card from a graveyard onto the battlefield under your control. You lose life equal to that card's mana value."]),
     ("Raise Dead", ["Return target creature card from your graveyard to your hand."])),
    (8, "bounce vs blink",
     ("Unsummon", "Return target creature to its owner's hand."),
     ("Vapor Snag", ["Return target creature to its owner's hand. Its controller loses 1 life."]),
     ("Cloudshift", ["Exile target creature you control, then return that card to the battlefield under your control."])),
    (9, "ramp vs tutor",
     ("Rampant Growth", "Search your library for a basic land card, put that card onto the battlefield tapped, then shuffle."),
     ("Nature's Lore", ["Search your library for a Forest card, put that card onto the battlefield, then shuffle."]),
     ("Diabolic Tutor", ["Search your library for a card, put that card into your hand, then shuffle."])),
    (10, "spell vs ability counter",
     ("Cancel", "Counter target spell."),
     ("Mana Leak", ["Counter target spell unless its controller pays {3}."]),
     ("Stifle", ["Counter target activated or triggered ability. (Mana abilities can't be targeted.)"])),
    (11, "narrow counter",
     ("Dismiss", "Counter target spell."),
     ("Exclude", ["Counter target creature spell.", "Draw a card."]),
     ("Stifle", ["Counter target activated or triggered ability. (Mana abilities can't be targeted.)"])),
    (12, "rhystic punisher",
     ("Rhystic Study", "Whenever an opponent casts a spell, you may draw a card unless that player pays {1}."),
     ("Mystic Remora", ["Cumulative upkeep {1} (At the beginning of your upkeep, put an age counter on this permanent, then sacrifice it unless you pay its upkeep cost for each age counter on it.)", "Whenever an opponent casts a noncreature spell, you may draw a card unless that player pays {4}."]),
     ("Forced Fruition", ["Whenever an opponent casts a spell, that player draws seven cards."])),
    (13, "numbers vs verbs",
     ("Divination", "Draw two cards."),
     ("Concentrate", ["Draw three cards."]),
     ("Mind Rot", ["Target player discards two cards."])),
    (14, "one vs all",
     ("Murder", "Destroy target creature."),
     ("Hero's Downfall", ["Destroy target creature or planeswalker."]),
     ("Day of Judgment", ["Destroy all creatures."])),
    (15, "counter polarity",
     ("Battlegrowth", "Put a +1/+1 counter on target creature."),
     ("Increasing Savagery", ["Put five +1/+1 counters on target creature. If this spell was cast from a graveyard, put ten +1/+1 counters on that creature instead."]),
     ("Grim Affliction", ["Put a -1/-1 counter on target creature, then proliferate. (Choose any number of permanents and/or players, then give each another counter of each kind already there.)"])),
    (16, "hug vs punish",
     ("Howling Mine", "At the beginning of each player's draw step, if this artifact is untapped, that player draws an additional card."),
     ("Font of Mythos", ["At the beginning of each player's draw step, that player draws two additional cards."]),
     ("Underworld Dreams", ["Whenever an opponent draws a card, this enchantment deals 1 damage to that player."])),
    (17, "mill vs draw",
     ("Glimpse the Unthinkable", "Target player mills ten cards."),
     ("Tome Scour", ["Target player mills five cards."]),
     ("Concentrate", ["Draw three cards."])),
    (18, "sac payload (freebie)",
     ("Village Rites", "Draw two cards."),
     ("Altar's Reap", ["Draw two cards."]),
     ("Bone Splinters", ["Destroy target creature."])),
    (19, "block vs attack",
     ("Bedlam", "Creatures can't block."),
     ("Falter", ["Creatures without flying can't block this turn."]),
     ("Peacekeeper", ["At the beginning of your upkeep, sacrifice this creature unless you pay {1}{W}.", "Creatures can't attack."])),
    (20, "hexproof vs shroud (domain)",
     ("Slippery Bogle", "Hexproof (This creature can't be the target of spells or abilities your opponents control.)"),
     ("Deadly Insect", ["Shroud (This creature can't be the target of spells or abilities.)"]),
     ("Boggart Brute", ["Menace (This creature can't be blocked except by two or more creatures.)"])),
    (21, "doubling domains",
     ("Parallel Lives", "If an effect would create one or more tokens under your control, it creates twice that many of those tokens instead."),
     ("Mondrak, Glory Dominus", ["If one or more tokens would be created under your control, twice that many of those tokens are created instead."]),
     ("Dictate of the Twin Gods", ["Flash", "If a source would deal damage to a permanent or player, it deals double that damage to that permanent or player instead."])),
    (22, "extra vs end turn",
     ("Time Warp", "Target player takes an extra turn after this one."),
     ("Temporal Manipulation", ["Take an extra turn after this one."]),
     ("Time Stop", ["End the turn. (Exile all spells and abilities, including this spell. The player whose turn it is discards down to their maximum hand size. Damage heals and \"this turn\" and \"until end of turn\" effects end.)"])),
    (23, "retrieve vs grave hate",
     ("Regrowth", "Return target card from your graveyard to your hand."),
     ("Eternal Witness", ["When this creature enters, you may return target card from your graveyard to your hand."]),
     ("Coffin Purge", ["Exile target card from a graveyard.", "Flashback {B} (You may cast this card from your graveyard for its flashback cost. Then exile it.)"])),
    (24, "counter vs uncounterable",
     ("Counterspell", "Counter target spell."),
     ("Negate", ["Counter target noncreature spell."]),
     ("Prowling Serpopard", ["This spell can't be countered.", "Creature spells you control can't be countered."])),
    (25, "fog vs anti-fog",
     ("Fog", "Prevent all combat damage that would be dealt this turn."),
     ("Ethereal Haze", ["Prevent all damage that would be dealt by creatures this turn."]),
     ("Skullcrack", ["Players can't gain life this turn. Damage can't be prevented this turn. Skullcrack deals 3 damage to target player or planeswalker."])),
    (26, "enemy vs self discard",
     ("Mind Rot", "Target player discards two cards."),
     ("Hymn to Tourach", ["Target player discards two cards at random."]),
     ("Careful Study", ["Draw two cards, then discard two cards."])),
    #round 3, from user reports (verified against the database 2026-07-15).
    #the current model is expected to fail most of these - they exist to
    #grade the next training pass
    (27, "flavour prefix trap",
     ("Farideh's Fireball", "1—9 | Farideh's Fireball deals 2 damage to each player."),
     ("Flame Rift", ["Flame Rift deals 4 damage to each player."]),
     ("Thunderwave", ["Roll a d20.", "1—9 | Thunderwave deals 3 damage to each creature.",
                      "10—19 | You may choose a creature. Thunderwave deals 3 damage to each creature not chosen this way.",
                      "20 | Thunderwave deals 6 damage to each creature your opponents control."])),
    (28, "mana colour is payload",
     ("Sol Ring", "{T}: Add {C}{C}."),
     ("Mind Stone", ["{T}: Add {C}.", "{1}, {T}, Sacrifice this artifact: Draw a card."]),
     #the abomination back face carries a literal "{T}: Add {C}{C}." - thats
     #what the site matched at 100% in the original report. left out here on
     #purpose: this exam judges lines, and the lesson is the front face
     ("Ulvenwald Captive // Ulvenwald Abomination", ["{T}: Add {G}.", "Defender", "{5}{G}{G}: Transform this creature."])),
    (29, "protection plus haste",
     ("Swiftfoot Boots", "Equipped creature has hexproof and haste. (It can't be the target of spells or abilities your opponents control. It can attack and {T} no matter when it came under your control.)"),
     ("Lightning Greaves", ["Equipped creature has haste and shroud. (It can't be the target of spells or abilities.)", "Equip {0}"]),
     ("Ring of Valkas", ["Equipped creature has haste. (It can attack and {T} no matter when it came under your control.)",
                         "At the beginning of your upkeep, put a +1/+1 counter on equipped creature if it's red.",
                         "Equip {1} ({1}: Attach to target creature you control. Equip only as a sorcery.)"])),
    (30, "all your spells vs this spell",
     ("Omniscience", "You may cast spells from your hand without paying their mana costs."),
     ("Dracogenesis", ["You may cast Dragon spells without paying their mana costs."]),
     ("Fierce Guardianship", ["If you control a commander, you may cast this spell without paying its mana cost.",
                              "Counter target noncreature spell."])),
    (31, "may vs can't",
     ("Omniscience", "You may cast spells from your hand without paying their mana costs."),
     ("Dracogenesis", ["You may cast Dragon spells without paying their mana costs."]),
     ("Hogaak, Arisen Necropolis", ["You can't spend mana to cast this spell.",
                                    "You may cast this card from your graveyard.",
                                    "Convoke, delve (Each creature you tap while casting this spell pays for {1} or one mana of that creature's color. Each card you exile from your graveyard pays for {1}.)",
                                    "Trample"])),
]


def cleaned(card_name, line):
    out = clean_line(line, card_name)
    if not out:
        raise ValueError("line cleaned away to nothing: " + card_name + " / " + line)
    return out


def run_model(label, model_id, prompt):
    from sentence_transformers import SentenceTransformer
    import torch

    print("")
    print("== " + label + " ==")
    print("loading " + model_id + " (downloads on first run)...")
    model = SentenceTransformer(model_id, device="cpu",
                                model_kwargs={"torch_dtype": torch.float32})

    #every unique cleaned line embeds once
    texts = []
    for num, name, anchor, pos, neg in TRIPLETS:
        texts.append(cleaned(anchor[0], anchor[1]))
        for line in pos[1]:
            texts.append(cleaned(pos[0], line))
        for line in neg[1]:
            texts.append(cleaned(neg[0], line))
    uniq = sorted(set(texts))
    kwargs = {"batch_size": 16, "normalize_embeddings": True, "convert_to_numpy": True}
    if prompt:
        kwargs["prompt"] = prompt
    embs = model.encode(uniq, **kwargs)
    vec = {}
    for i, t in enumerate(uniq):
        vec[t] = embs[i]

    results = []
    passes = 0
    for num, name, anchor, pos, neg in TRIPLETS:
        a = vec[cleaned(anchor[0], anchor[1])]
        best_pos = max(float(a @ vec[cleaned(pos[0], line)]) for line in pos[1])
        best_neg = max(float(a @ vec[cleaned(neg[0], line)]) for line in neg[1])
        ok = best_pos > best_neg
        if ok:
            passes += 1
        margin = best_pos - best_neg
        print(("%2d %-28s %s  match %5.1f%%  not %5.1f%%  (%+.1f)")
              % (num, name, "PASS" if ok else "FAIL", best_pos * 100, best_neg * 100, margin * 100))
        results.append((num, name, best_pos, best_neg, ok))

    print("score: " + str(passes) + "/" + str(len(TRIPLETS)))

    del model
    gc.collect()
    return passes, results


def main():
    scoreboard = []
    rows = []
    for label, model_id, prompt in MODELS:
        try:
            passes, results = run_model(label, model_id, prompt)
        except Exception as e:
            msg = str(e).replace("\n", " ")[:200]
            print("")
            print("== " + label + " ==")
            print("SKIPPED: " + msg)
            if "gated" in msg.lower() or "401" in msg or "403" in msg:
                print("(this repo is gated: accept the license on huggingface.co/" + model_id)
                print(" then run: huggingface-cli login)")
            scoreboard.append((label, None))
            continue
        scoreboard.append((label, passes))
        for num, name, best_pos, best_neg, ok in results:
            rows.append([label, num, name, round(best_pos, 4), round(best_neg, 4), "pass" if ok else "fail"])

    out_path = os.path.join(os.path.dirname(__file__), "bakeoff_results.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "triplet", "label", "best_match_sim", "best_not_sim", "result"])
        w.writerows(rows)

    print("")
    print("==== scoreboard ====")
    for label, passes in scoreboard:
        if passes is None:
            print("%-30s skipped" % label)
        else:
            print("%-30s %d/%d" % (label, passes, len(TRIPLETS)))
    print("")
    print("details written to " + out_path)


if __name__ == "__main__":
    main()
