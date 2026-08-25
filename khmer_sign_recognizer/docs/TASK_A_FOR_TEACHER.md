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

**What we found: yes, it matters, and unevenly.** Every condition costs
something, but not equally. The report names which change is hardest for this
dataset and by how many points.

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

The comparison table in the results shows what it actually bought us.

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
python algo_comparison/run_var_experiment.py --lang khmer_var
```
