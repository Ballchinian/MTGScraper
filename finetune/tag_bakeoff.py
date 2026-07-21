#which base model starts highest on the LINE -> TAG objective, before any
#fine tuning. the original bakeoff.py asks a different question (26 triplets,
#does this line mean the same as that line) and picked EmbeddingGemma for it.
#the objective has changed, so the choice of base should be re-measured rather
#than inherited: a model that is good at paraphrase is not automatically good
#at "what is this line about".
#
#run from the repo root, no database needed, everything comes out of traindata:
#    python finetune/tag_bakeoff.py
#every model is scored on the same held out cards and the same tag pool as
#finetune/tag_eval.py, so the numbers sit next to that harness's.
#
#IT IS THE TEXT SCORER. lines are ranked against the WORDS of each tag
#("slug: description"), which is exactly what training will optimise. that also
#means these numbers are not comparable to tag_eval.py's headline, which
#defaults to the centroid scorer: the fine tuned model in production has never
#been taught what a tag slug says, so text retrieval would flatter a stock
#model against it. compare bases with bases.
#
#the prompts matter and are not cosmetic. embeddinggemma and bge want a
#retrieval instruction, and scoring them bare understates them by a lot. each
#model is asked what prompts it ships with and given its own, which is the only
#fair way to line them up. the prompt used is printed with every score.

import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "traindata")
sys.path.insert(0, HERE)

#the five from the original bake-off, plus the tuned model when it is reachable.
#all of these were already pulled once for that bake-off, so a rerun is offline
MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "Alibaba-NLP/gte-modernbert-base",
    "google/embeddinggemma-300m",
    "Qwen/Qwen3-Embedding-0.6B",
]

KS = (1, 5, 10)


def load_pool():
    #the trainable pool IS whatever train_tags.jsonl contains: gen_training.py
    #already applied the AUC and the review before writing it, so reading the
    #file back needs neither tag_learnability.json nor gen_tag_review.py. that
    #keeps this runnable from three data files and two scripts, which matters
    #when the whole lot has to reach a colab box
    text = {}
    for line in open(os.path.join(DATA_DIR, "train_tags.jsonl"), encoding="utf-8"):
        p = json.loads(line)["positive"]
        text[p.split(":")[0]] = p
    tags = sorted(text)
    held = [json.loads(l) for l in open(os.path.join(DATA_DIR, "tag_testset.jsonl"), encoding="utf-8")]
    rows, golds = [], []
    for h in held:
        gold = set(h["tags"]) & set(tags)
        if gold:
            rows.append(h["line"])
            golds.append(gold)
    return tags, [text[t] for t in tags], rows, golds


def prompts_for(model):
    #a model's own prompts, if it ships any. sentence-transformers exposes them
    #as a dict on the model; the usual names are query/document for asymmetric
    #retrieval. anything else gets nothing, which is correct for a symmetric
    #model rather than a handicap
    have = getattr(model, "prompts", None) or {}
    q = d = None
    for name in ("query", "search_query", "Retrieval-query", "STS"):
        if name in have:
            q = name
            break
    for name in ("document", "passage", "search_document", "Retrieval-document"):
        if name in have:
            d = name
            break
    return q, d


def score(sims, golds, tags):
    ix = {t: i for i, t in enumerate(tags)}
    order = np.argsort(-sims, axis=1)
    rec = {k: 0.0 for k in KS}
    ap = 0.0
    for row, gold in enumerate(golds):
        ranked = [tags[i] for i in order[row]]
        for k in KS:
            rec[k] += len(set(ranked[:k]) & gold) / len(gold)
        hits = [i for i, t in enumerate(ranked) if t in gold]
        ap += sum((j + 1) / (r + 1) for j, r in enumerate(hits)) / len(gold)
    n = len(golds)
    return {"r1": rec[1] / n, "r5": rec[5] / n, "r10": rec[10] / n, "map": ap / n}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--models", default=None, help="comma separated, defaults to the bake-off five")
    ap_.add_argument("--batch", type=int, default=32)
    args = ap_.parse_args()
    names = args.models.split(",") if args.models else MODELS

    tags, tag_texts, lines, golds = load_pool()
    print(str(len(lines)) + " held out lines, " + str(len(tags)) + " tags to rank\n")

    import torch
    from sentence_transformers import SentenceTransformer

    results = []
    for name in names:
        print("=" * 70)
        print(name)
        try:
            model = SentenceTransformer(name, model_kwargs={"torch_dtype": torch.float32},
                                        trust_remote_code=True)
        except Exception as e:
            print("  could not load: " + str(e)[:120])
            continue
        q, d = prompts_for(model)
        print("  prompts: query=" + str(q) + " document=" + str(d))
        kw_q = {"prompt_name": q} if q else {}
        kw_d = {"prompt_name": d} if d else {}
        L = model.encode(lines, batch_size=args.batch, normalize_embeddings=True,
                         show_progress_bar=False, **kw_q)
        T = model.encode(tag_texts, batch_size=args.batch, normalize_embeddings=True,
                         show_progress_bar=False, **kw_d)
        r = score(np.asarray(L, dtype=np.float32) @ np.asarray(T, dtype=np.float32).T, golds, tags)
        r["name"] = name
        r["prompt"] = "own" if (q or d) else "none"
        results.append(r)
        print("  recall @1 %5.1f%%   @5 %5.1f%%   @10 %5.1f%%   MAP %5.1f%%"
              % (100 * r["r1"], 100 * r["r5"], 100 * r["r10"], 100 * r["map"]))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("BASE MODELS ON THE TAG OBJECTIVE, best recall @10 first")
    print("%-42s %8s %8s %8s" % ("model", "r@10", "MAP", "prompt"))
    for r in sorted(results, key=lambda x: -x["r10"]):
        print("%-42s %7.1f%% %7.1f%% %8s"
              % (r["name"][:42], 100 * r["r10"], 100 * r["map"], r["prompt"]))
    print("\nzero shot only. the best starting point is not guaranteed to be the")
    print("best finished model, but it is the only evidence available before a run.")


if __name__ == "__main__":
    main()
