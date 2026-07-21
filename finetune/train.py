#fine tunes an embedding model on the generated mtg training data.
#
#TWO OBJECTIVES LIVE HERE, pick with --objective:
#   lines  the original: what the site shipped. lines that mean the same thing
#          land close together, and attribution recovers tags from that
#          indirectly, which is what its 88%/82% costs
#   tags   the retarget: a line lands close to the TEXT OF ITS TAGS. trained on
#          the single-line cards where a human's tag belongs to that one line
#          with no inference. this is the one meant to replace the other
#   both   mix them, an experiment rather than a plan
#
#the exam triplets only appear in here as a held out evaluator, never as
#training data. the line objective mixes three lessons with three losses:
#   pairs (mined variants + keyword meanings + rewordings) -> pull together
#   triplets (anchor, variant, flipped anchor)             -> pull + push apart
#   labeled flips (anchor vs flipped, label 0)             -> push apart hard
#
#under --objective tags the 26 bakeoff triplets stop being the target and
#become a regression guard. the number that matters is recall @10 on the held
#out cards, printed here during training and measured properly afterwards by
#    python -m finetune.tag_eval
#which is the real judge, the same way bakeoff.py is for the line objective.

import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

random.seed(7)
HERE = os.path.dirname(os.path.abspath(__file__))

#the rare classes that matter most (the loot order flip especially) get
#oversampled, the bloated ones get capped so enters/dies and may/can't dont
#drown everything else out. the round-3 classes are all rare: they exist
#because users caught the model failing them
CAP = 1500
OVERSAMPLE = {"loot order flip": 5, "attack/block flip": 2, "hand/battlefield flip": 2,
              "self/general flip": 5, "mana amount variant": 5, "subtype variant": 3,
              "etb wrapper": 2,
              #round 4, from the 2026-07-19 ranking audit rather than user
              #reports. both mine rare, and both answer a failure sitting at
              #rank 1 or filling half a top 20, so they get lifted into the
              #same 500-800 band the loot order flip sits in
              "restriction target flip": 5, "toughness null": 2,
              #the qualifier class mines only 179 after the guards that keep it
              #off pairs of removal spells, but it answers 8% of the harvested
              #false positives, so it gets lifted alongside them. its partner
              #"same trigger, different effect" needs no lift, it hits the cap
              "same opening, conflicting qualifier": 3}

#the tag objective needs its own flattening, for a different reason. seven tags
#carry 23% of the 36k pairs (activated-ability alone is 5.1%, 1843 of them), so
#without a cap the model spends a quarter of its time on a handful of labels
#that say almost nothing about a card. the tail is the opposite shape: 296 of
#the 654 tags have under 20 pairs and every one of those is precious
TAG_CAP = 300


def load_tag_pairs():
    #(line, "slug: description") from finetune/traindata/train_tags.jsonl, the
    #single-line cards where a human's tag can only belong to the one line
    rows = load_jsonl("train_tags.jsonl")
    by_tag = {}
    for r in rows:
        by_tag.setdefault(r["positive"], []).append(r)
    out = []
    for tag, group in sorted(by_tag.items()):
        random.shuffle(group)
        out.extend(group[:TAG_CAP])
    random.shuffle(out)
    capped = sum(1 for g in by_tag.values() if len(g) > TAG_CAP)
    print("  %d pairs over %d tags, %d tags capped at %d"
          % (len(out), len(by_tag), capped, TAG_CAP))
    return out


def load_jsonl(name):
    path = os.path.join(HERE, "traindata", name)
    if not os.path.exists(path):
        print("missing " + name + ", skipping it")
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def balance(rows):
    by_why = {}
    for r in rows:
        why = r.get("why", "?").replace(" (real line)", "")
        by_why.setdefault(why, []).append(r)
    out = []
    for why, group in by_why.items():
        random.shuffle(group)
        group = group[:CAP] * OVERSAMPLE.get(why, 1)
        out.extend(group)
        print("  %-24s %d" % (why, len(group)))
    random.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    #the tuned line-to-line model, not a stock one. decided 2026-07-21: it is
    #already 768 dims so nothing downstream moves, it already knows tap from
    #untap and the rest of the vocabulary bakeoff.py tests, and keeping that
    #knowledge is what lets the guards detect the umbrella-tag problem rather
    #than just measuring a model relearning magic from scratch.
    #
    #the tag bakeoff put five stock bases inside a 2.5 point band zero shot,
    #which is not a ranking, and the leader was 384 dims: choosing it would
    #mean changing EMBED_DIMS, the column type and the hnsw index to chase a
    #lead that almost certainly does not survive fine tuning.
    #
    #for a laptop smoke test pass --model sentence-transformers/all-MiniLM-L6-v2
    ap.add_argument("--model", default="BallchinianMan/mtg-tuned-embeddinggemma-300m")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=0, help="0 picks one based on model size")
    ap.add_argument("--objective", default="tags", choices=["tags", "lines", "both"],
                    help="tags is the retarget, lines is what the site shipped")
    ap.add_argument("--limit", type=int, default=0,
                    help="train on this many rows per dataset, for smoke testing the wiring")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss, ContrastiveLoss
    from sentence_transformers.evaluation import TripletEvaluator, InformationRetrievalEvaluator
    from sentence_transformers.training_args import BatchSamplers
    from bakeoff import TRIPLETS
    from common.cards import clean_line

    is_gemma = "gemma" in args.model.lower()
    #embeddinggemma is trained to see a task prompt, so train and eval with the
    #same one. the tuned model must then be used with this prefix forever
    prefix = "task: sentence similarity | query: " if is_gemma else ""
    batch = args.batch or (16 if is_gemma else 64)

    print("loading data...")
    train_sets, losses = {}, {}

    if args.objective in ("tags", "both"):
        print("line -> tag pairs:")
        tag_pairs = load_tag_pairs()
        if not tag_pairs:
            print("no train_tags.jsonl. build it with:")
            print("  python finetune/gen_training.py --tags-only")
            sys.exit(1)
        train_sets["tags"] = Dataset.from_dict({
            "anchor": [prefix + r["anchor"] for r in tag_pairs],
            "positive": [prefix + r["positive"] for r in tag_pairs],
        })

        #the near misses, as explicit negatives rather than whatever the batch
        #happened to contain: a line tagged attack-trigger is NOT block-trigger
        #or death-trigger, one tagged discard is NOT random-discard. these are
        #the distinctions in-batch negatives are least likely to serve up,
        #because a random other row is usually nothing like the anchor
        tag_trips = load_jsonl("train_tag_triplets.jsonl")
        if tag_trips:
            print("  plus " + str(len(tag_trips)) + " sibling triplets")
            train_sets["tag_triplets"] = Dataset.from_dict({
                "anchor": [prefix + r["anchor"] for r in tag_trips],
                "positive": [prefix + r["positive"] for r in tag_trips],
                "negative": [prefix + r["negative"] for r in tag_trips],
            })

    if args.objective in ("lines", "both"):
        pairs = (load_jsonl("train_pairs.jsonl") + load_jsonl("train_keywords.jsonl")
                 + load_jsonl("train_rewordings.jsonl") + load_jsonl("train_retemplate.jsonl"))
        negatives = load_jsonl("train_negatives.jsonl")
        triplets = load_jsonl("train_triplets.jsonl")

        print("positives by class:")
        pairs = balance(pairs)
        print("negatives by class:")
        negatives = balance(negatives)

        train_sets["pairs"] = Dataset.from_dict({
            "anchor": [prefix + r["anchor"] for r in pairs],
            "positive": [prefix + r["positive"] for r in pairs],
        })
        train_sets["triplets"] = Dataset.from_dict({
            "anchor": [prefix + r["anchor"] for r in triplets],
            "positive": [prefix + r["positive"] for r in triplets],
            "negative": [prefix + r["negative"] for r in triplets],
        })
        #labeled pairs for the contrastive loss: every flip is a 0, and an equal
        #helping of positives are 1s so the loss sees both sides
        ones = random.sample(pairs, min(len(pairs), len(negatives)))
        train_sets["labeled"] = Dataset.from_dict({
            "sentence1": [prefix + r["anchor"] for r in negatives] + [prefix + r["anchor"] for r in ones],
            "sentence2": [prefix + r["negative"] for r in negatives] + [prefix + r["positive"] for r in ones],
            "label": [0] * len(negatives) + [1] * len(ones),
        })

    #the 26 exam triplets, held out from all training data. under the line
    #objective this is the target; under the tag objective it is a REGRESSION
    #GUARD, there to catch the retrain forgetting what lines mean, not to be
    #maximised. the real judge stays bakeoff.py either way
    ev_a, ev_p, ev_n = [], [], []
    for num, name, anchor, pos, neg in TRIPLETS:
        ev_a.append(prefix + clean_line(anchor[1], anchor[0]))
        ev_p.append(prefix + clean_line(pos[1][0], pos[0]))
        ev_n.append(prefix + clean_line(neg[1][0], neg[0]))
    evaluator = TripletEvaluator(anchors=ev_a, positives=ev_p, negatives=ev_n, name="exam")

    #and the number that actually decides the retrain: given a held out line and
    #every trainable tag, are the right tags in its top ten? this is the ship
    #bar in tag_eval.py, computed here so a run reports it as it goes rather
    #than only after the upload. the tag texts come out of the training file
    #so this needs no database
    tag_ir = None
    if args.objective in ("tags", "both"):
        held = load_jsonl("tag_testset.jsonl")
        tag_text = {r["positive"].split(":")[0]: r["positive"] for r in load_jsonl("train_tags.jsonl")}
        corpus = {slug: prefix + text for slug, text in tag_text.items()}
        queries, relevant = {}, {}
        for i, h in enumerate(held):
            gold = {t for t in h["tags"] if t in corpus}
            if not gold:
                continue
            queries["q" + str(i)] = prefix + h["line"]
            relevant["q" + str(i)] = gold
        if queries:
            tag_ir = InformationRetrievalEvaluator(
                queries=queries, corpus=corpus, relevant_docs=relevant,
                accuracy_at_k=[1, 10], precision_recall_at_k=[1, 10], map_at_k=[10],
                name="tags", show_progress_bar=False)
            print("tag retrieval evaluator: " + str(len(queries)) + " held out lines against "
                  + str(len(corpus)) + " tags")

    #a real run is hours on a gpu box, so --limit exists to prove the wiring
    #end to end in minutes on a laptop before spending any of that
    if args.limit:
        train_sets = {k: v.select(range(min(args.limit, len(v)))) for k, v in train_sets.items()}
        print("LIMIT: " + str(args.limit) + " rows per dataset, this is a smoke test not a run")

    print("loading " + args.model + "...")
    model = SentenceTransformer(args.model, model_kwargs={"torch_dtype": torch.float32})
    print("exam before training:", evaluator(model))
    if tag_ir:
        print("tags before training:", tag_ir(model))

    out_dir = os.path.join(HERE, "mtg-tuned-" + args.model.split("/")[-1])
    loss_mnrl = MultipleNegativesRankingLoss(model)
    loss_contrastive = ContrastiveLoss(model)
    for name in train_sets:
        losses[name] = loss_contrastive if name == "labeled" else loss_mnrl

    train_args = SentenceTransformerTrainingArguments(
        output_dir=out_dir + "-checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available() and not is_gemma,  #T4 has no bf16, keep gemma in fp32
        eval_strategy="no",
        save_strategy="no",
        logging_steps=100,
        report_to=[],
        #THE LOAD BEARING LINE FOR THE TAG OBJECTIVE. MultipleNegativesRanking
        #treats every other row's positive in a batch as this row's negative,
        #which is fine when positives are unique and actively wrong here: the
        #same line appears 4.3 times carrying different true tags, and
        #activated-ability is 5.1% of the pool, so a batch of 64 expects around
        #three copies of it. left alone the loss spends its time pushing a line
        #away from tags it genuinely has. NO_DUPLICATES builds batches with no
        #repeated value in any column, which kills the worst of it (the same
        #tag text can no longer sit in a batch as one row's positive and
        #another's negative).
        #
        #what it does NOT kill: line A batched with a row whose positive is a
        #DIFFERENT tag A also carries. that pair is still a false negative,
        #it is just rarer (654 tags against batches of 16 to 64). if the
        #retrain plateaus below the bar, masking those with the (line, tag)
        #truth table from train_tags.jsonl is the next lever to pull
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_sets,
        loss=losses,
    )
    trainer.train()

    print("exam after training:", evaluator(model))
    if tag_ir:
        print("tags after training:", tag_ir(model))
    model.save_pretrained(out_dir)
    print("\nsaved to " + out_dir)
    if args.objective == "lines":
        print("next: copy that folder into finetune/ on your machine, then add")
        print('  ("mtg-tuned", r"' + out_dir + '", ' + (('"' + prefix + '"') if prefix else "None") + "),")
        print("to MODELS in bakeoff.py and rerun it for the real per-triplet exam.")
    else:
        print("next, in this order:")
        print("  1. upload to a NEW hugging face repo. the old one is the rollback,")
        print("     do not overwrite it")
        print("  2. point EMBED_MODEL at it and run the update workflow against a COPY")
        print("     of the database, never production: the write destroys the old")
        print("     vectors in place and they cannot be recomputed without a rerun")
        print("  3. python -m finetune.tag_eval, which is the real judge. the ship bar")
        print("     is recall @10 at 95%, and the current model sits at 47.0%")
        print("  4. python finetune/bakeoff.py as a regression guard, NOT a target")


if __name__ == "__main__":
    main()
