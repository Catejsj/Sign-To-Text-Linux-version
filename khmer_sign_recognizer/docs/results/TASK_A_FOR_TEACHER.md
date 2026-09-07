# Task A — Explaining It To The Teacher

Everything below is in the order you'd say it. Each part has **what we did**,
**what we found**, and **what to say**.

---

## Open with this

> We recorded 7 Khmer signs, 12 times each, from 4 different people — 337
> recordings. We tried 9 machine-learning algorithms on exactly the same data
> and compared them properly. We also designed the recording so we could test
> whether the room conditions — lighting, distance, standing position — change
> how well it works.

Two sentences. Everything else is detail.

---

## 1. Why we don't use images

The most common question, so answer it before it's asked.

A camera watches the signer, but **we throw the picture away immediately** and
keep only the positions of 48 body points — shoulders, elbows, wrists, and 21
points on each hand — tracked across 60 frames (about two seconds).

So a recording is not a video. It's a list of numbers describing where the body
was at each moment.

**Why that's better:**

- The model can't cheat by learning a shirt colour, a face, or a bedroom wall.
  It only sees the shape the body made.
- It's tiny and fast — training takes seconds, not hours.
- It works the same in any room, because the room was never recorded.

Then we compress each recording into **576 numbers**: for every joint we keep
the average position, how much it moved, and the extremes it reached. That's
what every algorithm sees.

> **If asked "why not the full movement?"** — these summaries were enough to
> reach the high seventies with simple, explainable algorithms. Using the full
> sequence needs a neural network, which is much harder to justify on 337
> recordings.

---

## 2. How we tested fairly

This is the part that makes it an experiment rather than a demo, so don't rush
it.

You cannot test a model on the recordings it learned from — that's marking your
own homework. So we hold **25% back**, train on the rest, and test on the part
it has never seen.

Two details that matter:

**We split by take, not by individual sample.** When a recording is held back
for testing, everything derived from it is held back too. Otherwise a near-copy
of a test recording sits in the training set and the score becomes meaningless.

**We repeat it 8 times** with a different 25% held back each time, and average.
One split could be lucky; eight averages the luck out. The ± column in the
results is how much the score wobbled — small means stable.

> **Say this:** "We train on three quarters and test on the quarter the model has
> never seen, and we repeat that eight different ways so the result isn't a
> fluke."

---

## 3. The four numbers in the results table

| metric | plain meaning |
|---|---|
| **Accuracy** | out of 100 recordings, how many were correct |
| **Precision** | when it says "អរគុណ", how often is that true — measures **false alarms** |
| **Recall** | of all the real "អរគុណ" recordings, how many did it catch — measures **misses** |
| **macro-F1** | precision and recall combined; the headline number |

**Why precision and recall are separate.** A model can cheat on either one. If it
almost never guesses "អរគុណ" but is right the rare times it does, precision looks
excellent while recall is terrible — it's missing nearly all of them. F1 combines
them so neither can be gamed.

**Why "macro".** Every sign counts equally, regardless of how many recordings it
has. Without that, a model could ignore one sign entirely and still score well.

---

## 4. The result, and how to state it honestly

**Bagged Trees came first**, with Gradient Boosting close behind and Random
Forest next.

**Do not claim a single winner.** The top few are within one standard deviation
of each other — the gap between them is smaller than the natural wobble between
splits. Saying so is stronger than picking one, because it shows you understand
what the error bars mean.

What you **can** claim confidently is the gap between the leading group and the
bottom. k-NN is far below everything else, well outside any margin of error.

> **Say this:** "The top three are statistically tied — we can't separate them
> with this much data. What we can say is that the tree-based methods clearly
> beat the rest, and k-NN is unsuitable for this problem."

> **If asked why k-NN fails:** it classifies by finding the most similar
> recordings it has already seen. With 576 numbers describing each recording,
> "similar" stops being meaningful — everything ends up roughly equally far from
> everything else. It's a known weakness of distance-based methods when there
> are many features.

---

## 5. Did the room conditions matter?

**This is the most interesting section — spend your time here.**

**What we did.** The 12 recordings of each sign weren't identical. They follow a
deliberate grid: two lighting levels, two distances from the camera, three
standing positions.

That lets us ask a sharper question. Instead of hiding a random 25%, we hide **an
entire condition** — train only on brightly lit recordings, then test only on dim
ones. The model has never seen dim lighting. Does it still work?

**What we found: yes, and not where we expected.**

| trained on | tested on | macro-F1 |
|---|---|---|
| bright light | dim light | **78.6%** — barely any cost |
| near the camera | further away | **66.3%** — the hardest |
| middle/left | right | 66.9% |

**Distance and standing position hurt; lighting barely does.**

This is worth dwelling on, because when only **one** person had recorded, we got
the opposite answer — lighting was catastrophic then, dropping to 23%. Four
people recording in four different rooms gave the model enough natural variety
in lighting that it stopped mattering.

> **Say this:** "With one signer, lighting looked like our biggest problem. With
> four signers recording in four different rooms, it stopped being a problem at
> all — the variety came for free. What actually hurts is distance and where the
> person stands."

That is a real finding about data collection, not just about algorithms: **the
cheapest fix for a condition problem is more people, not more careful control of
the condition.**

**The point worth making:** a model can look excellent in the room it was
recorded in and get noticeably worse in a different one. Testing on a random
split alone would never reveal that, because the training and test recordings
would come from the same conditions.

> **Say this:** "We didn't just measure accuracy — we measured how much accuracy
> survives a change in the room. That's the number that matters if the system is
> ever used somewhere new."

**The honest caveat — say it yourself, don't wait to be asked.** We always
recorded the bright takes first and the dim ones afterwards. So "dim lighting"
and "recorded later, when the signer was tired" are tangled together. We can't
fully prove the drop is caused by the light rather than by drifting technique.
The fix is easy — have half the people record dim first — and we know to do that
next time.

Admitting this is a strength. It shows you know what a confounding variable is.

---

## 6. Real vs synthetic data (the extra experiment)

**What synthetic data is.** From one real recording we can generate extra
training examples by mathematically stretching the skeleton — same movement,
different body proportions. It's pure geometry on the numbers; no new recording,
no camera. Because we only scale bone lengths and never change joint angles, the
sign itself is provably unchanged.

**Why we tried it.** More training data usually helps, and this costs nothing.

**How we tested it fairly.** Two rules, both essential:

1. **We only ever test on real recordings.** Scoring a model on synthetic data
   would be measuring whether it can recognise our own maths.
2. **A synthetic copy stays with its parent.** If a real take is held out for
   testing, all of its synthetic children are held out too. Otherwise the model
   trains on near-copies of the test data and the score is inflated.

> **Say this:** "We generate extra training examples by stretching the skeleton
> to different body sizes. We never test on them, and we make sure a recording's
> synthetic copies never end up on the opposite side of the split from the
> original."

### What we found — and it's the most interesting result we have

Synthetic data helped **enormously** here. Random Forest went from **79.8% to
96.2%** macro-F1; the weakest algorithm, k-NN, went from 45% to 93%.

But on our *other* dataset, the same test showed synthetic making things very
slightly **worse** — 97.3% down to 96.4%.

Both are correct. The difference is how much real data each started with:

| dataset | recordings per sign | real only | + synthetic |
|---|---|---|---|
| Task A grid | 12 | 79.8% | **96.2%** |
| our larger corpus | 30 | 97.3% | 96.4% |

**Notice both end up at about 96%.** With 12 recordings per sign the model is
starved, and synthetic data buys back almost exactly what wasn't recorded. With
30 per sign there was already enough, so the synthetic adds nothing and costs a
little precision.

> **Say this:** "Synthetic data is worth a lot when you don't have much real
> data, and worth nothing once you do. On the 12-recordings-per-sign set it
> gained 16 points; on the 30-per-sign set it lost one. Both datasets end at
> about 96% either way — it buys back the recordings we didn't make, up to a
> ceiling."

That's a much better answer than "augmentation helps", because it says **when**
it helps and **why it stops**.

> **If asked "how do you know it isn't cheating?"** — two things. We never test
> on synthetic data. And a recording's synthetic copies always stay on the same
> side of the split as the original, so the model is never tested on a warped
> copy of something it trained on. We verified that directly: 337 groups, one
> real recording each, none appearing on both sides. We also reproduced the
> result with a second, independently written piece of code.

---

## 7. Limitations to state up front

- **337 recordings is a small dataset.** Big differences are trustworthy; a
  couple of points between neighbouring algorithms is not.
- **Lighting is confounded with recording order** (see Section 5).
- **The test set mixes the same 4 people who are in training.** So this measures
  recognition for people the model has already seen. Performance for a complete
  stranger is a separate question — you answer that by holding one person out
  entirely, which is what the other experiment does.

---

## 8. Likely questions

**"Which algorithm should we use?"**
Depends on the situation, and that's the finding. For a controlled setting, the
top scorer. If conditions vary, choose whichever held up best under the condition
tests — the report shows those can be different models.

**"Isn't that accuracy a bit low?"**
It's 7 signs, 337 recordings, simple algorithms, and no neural network. Random
guessing would be about 14%. The more meaningful number is how much accuracy
survives a change in conditions.

**"Why 12 recordings per sign?"**
Because 12 is exactly 2 lighting × 2 distances × 3 positions — one recording of
every combination. That's what makes the condition tests possible.

**"How do we know you didn't just get lucky?"**
Every algorithm has a fixed random seed and the whole thing regenerates with one
command, producing identical numbers on any machine. And every score is the
average of 8 different splits, reported with its standard deviation.

**"Could someone else reproduce this?"**
Yes — one command:
```
python algo_comparison/run_var_experiment.py (replaced by `run_task_a.py`) --lang khmer_var
```
