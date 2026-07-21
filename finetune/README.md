# The embedding model project

The site finds similar cards by turning every line of rules text into a list of
numbers (an "embedding") where lines that mean similar things land close
together. That trick has one famous weakness, and people found it on launch day:
the model reads sentences a bit like a bag of words. "Draw a card, then discard
a card" and "Discard a card: Draw a card" use the same words, so the model
called them 98% similar. Any Magic player knows one is card selection and the
other is a downside.

This folder is the project of fixing that: build a test, test every model on the
market, discover none of them fix it, teach one ourselves, prove it worked.

The punchline: the fine tuned model scores **25/26** on a test where the best
off-the-shelf models score 21/26, including both questions that every stock
model on the market failed.

## Where the data comes from, and what happens to it

Everything the site shows you is public data, and it is worth being plain about
which parts are borrowed and which are computed here.

| Source | What it is | What the site does with it |
| --- | --- | --- |
| [Scryfall](https://scryfall.com) bulk data | card names, rules text, type lines, images, prices, EDHREC ranks | downloaded once a day, stored, and displayed. prices are the cheapest paper printing, converted for the currency toggle |
| [Scryfall Tagger](https://tagger.scryfall.com) | community written tags describing what cards do | the concepts axis. a card's tags become a weighted vector, and two cards score on how much of that they share |
| this project | the embeddings, the line to tag attribution, the uniqueness scores | computed from the two above and stored alongside them |

The embeddings are the part that is genuinely ours. Every line of rules text is
cleaned (reminder text stripped, the card's own name swapped for "this card" so
names cannot influence matching), then run through the fine tuned model once, at
ingest time. Nothing is embedded while you search: the search compares numbers
that were computed days ago.

Three things the site does NOT do. It stores no accounts and no decklists. It
does not send your searches anywhere. And the tags are Tagger's work, not ours,
which is why the concepts axis links back to them rather than pretending it
invented the vocabulary.

## What is published here, and what is not

This README is the only file in this folder that ships. The scripts that mine
the training data, the training data itself, the hand judged tag verdicts and
the exams all stay local.

Not because the approach is a secret. It is described below in enough detail to
argue with. It is that the data is the actual work: every exam question was
verified card by card, every tag verdict was read and judged one at a time, and
that judgement is the only part of this that nobody else has. Publishing the
recipe is fine. Publishing the thing that took the evenings is not.

## Step 1: build an exam before shopping

You can't pick a model on vibes, and the public leaderboards test the wrong
thing (search engines, not "do these two abilities mean the same thing"). So the
first artifact is an exam: 26 hand-reviewed triplets, each one an anchor line, a
line that *should* match it, and a trap line that looks nearly identical but
means something else. Rummaging Goblin is Merfolk Looter's trap. Refocus (untap)
is Pressure Point's (tap), and they differ by two letters.

The review settled the site's philosophy of similarity in writing: **same
mechanism, flexible parameters**. Numbers, colors and riders are forgivable; a
flipped mechanism is not, even when the deck slot matches (Lightning Bolt and
Murder both kill a creature, and are still not "similar").

The exam never gets trained on. It exists to judge.

## Step 2: the bake-off

Five models, all 26 triplets, scored exactly the way the site itself ranks
things (a candidate card's best matching line counts).

| Model | Score | Notes |
| --- | --- | --- |
| all-MiniLM-L6-v2 (the site's model at the time) | 20/26 | rates the rummage trap 98.2% similar to Merfolk Looter |
| bge-small-en-v1.5 | 20/26 | the planned upgrade. identical score, same failures |
| gte-modernbert-base | 19/26 | worse than what we had |
| EmbeddingGemma-300m | 21/26 | only stock model to pass loot vs rummage, barely (+1.5) |
| Qwen3-Embedding-0.6B | 21/26 | scores compress into a 75-98% band that would break the match % display |

The finding that shaped everything after: **every model failed the same
questions.** Tap vs untap and "reanimate reworded" went 0 for 5. These models
were all trained on the same internet and learned the same habit of treating
word-swapped sentences as paraphrases.

## Step 3: write the textbook

Fine tuning means actually changing the model: showing it thousands of examples
of "these mean the same, move them together" and "these do not, push them apart"
until its internal weights shift. The exam is 26 questions; the textbook needs
thousands, and almost all of it is generated automatically from the site's own
distinct rules lines rather than written by hand.

Roughly, it teaches four things:

- **Parameters are flexible.** Real card lines that differ only by numbers,
  riders or scope. "Draw two cards" against "Draw three cards", Unsummon against
  Vapor Snag, Cancel against Negate.
- **Mechanisms are not.** Real lines with exactly one mechanism flipped: tap for
  untap, enters for dies, gain for lose, hand for battlefield, attack for block,
  draw-then-discard reversed. A good fraction of the flips turn out to be real
  printed cards, which is the best kind of lesson.
- **Function over phrasing.** Wizards renamed the same effects repeatedly over
  thirty years, so running those renames backwards over modern text produces
  same-meaning pairs whose authority is Wizards rather than a regex guess.
- **A shared clause must not swamp a differing one.** Of the false positives
  harvested from the site's own rankings where either side had a trigger
  condition at all, **77% shared the condition and differed in the effect**.
  "At the beginning of your upkeep" opens one card that exiles your library, one
  that sacrifices an Aura and one that adds a time counter, and the model called
  them alike.

Every line that appears in the exam is excluded from all of it, so passing the
exam can never be memorization.

## Step 4: the training

Runs on a free Colab GPU in under an hour. Three lessons with three matched
losses: pairs pull together, triplets pull and push at once, and the flips go
through a contrastive loss that explicitly pushes near-identical wordings apart,
which is the exact ability no stock model has. Rare but important classes get
oversampled and bloated ones get capped.

The base model is EmbeddingGemma-300m, the strongest small model from the
bake-off. It expects a task prompt, so the tuned model was trained with one and
must always be used with it.

## Step 5: the rematch

Same exam, same rules, tuned model added as the sixth contender:

| Model | Score |
| --- | --- |
| **mtg-tuned EmbeddingGemma** | **25/26** |
| EmbeddingGemma-300m / Qwen3-0.6B | 21/26 |
| all-MiniLM-L6-v2 / bge-small | 20/26 |

Both impossible questions fell: tap vs untap passed by +35 points, the reanimate
rewording by +31. The margins tell the real story. The stock models that passed
loot vs rummage did it by a fragile 1-2 points; the tuned model passes by +12.5,
and rates Blood Artist against Soul Warden (dies vs enters) as *negatively*
similar, which is the model saying "opposites" rather than "close call". One
question regressed, a fair trade for five fixes.

## Step 6: the sanity check

25/26 on 26 questions could still hide a model that went weird everywhere else,
so the last step embeds the entire database with both the tuned model and the
old one and prints the top matches for 14 ordinary searches side by side. The
neighborhoods are sane, and the report catches the old model doing the launch
day complaint in the wild: its #1 match for Merfolk Looter's ability is the
rummage line, at 98.2%. The tuned model's top matches are all true looting
variants. Same story for lifegain, reanimation and discard.

## What is being worked on now

The model above is good at "these two lines mean the same thing". That turns out
to be the wrong target for where the site is going.

The more useful question is "what is this line ABOUT", because that is what lets
you pick one ability on a card and browse outward from it, which is the one thing
neither Scryfall nor Tagger can do: both are card-level, and a card tagged three
ways gives you no way to know which tag belongs to which ability.

So the current work retrains on (line, tag) pairs instead of (line, line) pairs,
using the cards whose entire rules text is a single line. On those, every tag a
human typed belongs to that one line with no inference required, which is a large
amount of free supervision that nobody had to label. The exam for it is the same
shape as the one above: hold cards back, and ask whether the model can rank the
right tags first for text it has never seen.
