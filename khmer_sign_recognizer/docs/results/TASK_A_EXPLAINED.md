# Task A Report — What It Actually Says

A plain-language companion to `Task_A_Report.docx`, section by section, so you
can explain it out loud. Each part below says **what it is**, **what the numbers
mean**, and **what to say** if someone asks.

---

## The one-sentence version

> We recorded 7 Khmer signs 12 times each, tried 9 different machine-learning
> algorithms on the same data, and measured which recognises the signs best.
> The best got about 91% correct. We also tested whether changing the room —
> lighting, distance, where you stand — affects it, and lighting matters a lot
> while the others barely matter.

If you only remember one thing, remember that.

---

## Background you need before Section 1

**We don't feed the computer video.** That surprises people, so say it early.

A camera watches you sign, but we immediately throw the picture away and keep
only the **positions of your joints** — shoulders, elbows, wrists, and 21 points
on each hand. 48 points in total, tracked across 60 frames (about two seconds).

So one recording is just a list of numbers: where every joint was, at every
moment.

**Why that's better than using the image:** the computer can't accidentally learn
your shirt colour, your face, or your bedroom wall. It only sees the shape your
body made. It also means the file is tiny and training takes seconds instead of
hours.

**One more step.** 48 joints × 60 frames is a lot of numbers, and most algorithms
want a fixed-size list. So for each joint we compute four summary numbers over
the whole clip:

| summary | what it captures |
|---|---|
| **mean** | where that joint usually was |
| **standard deviation** | how much it moved |
| **minimum** | the furthest it went one way |
| **maximum** | the furthest it went the other way |

That gives **576 numbers per recording**, and that's what every algorithm sees.

> If asked "why not use the whole movement?": because these summaries were enough
> to reach ~91%, and they let us use simple, explainable algorithms. Using the
> full sequence needs a neural network, which is harder to justify and harder to
> train on 84 recordings.

---

## Section 1 — Data and method

**What it is:** the setup. 7 signs, 12 recordings of each, 84 in total, one
person.

**The key idea here is the split.** You cannot test a model on the same
recordings it learned from — it would just be recalling them, like marking your
own homework. So we hold some back:

- **75%** of recordings → the model learns from these
- **25%** → hidden away, used only to test

**"Split by take, not by sample"** — the phrase in the report. It means when a
recording is held back for testing, *everything derived from it* is held back
too. If even one copy of a test recording leaked into training, the score would
look great and mean nothing.

**"Averaged over 8 random splits"** — we do the whole thing 8 times with a
different 25% hidden each time, and average. One split could be lucky. Eight
splits average the luck out.

> **Say this:** "We train on three quarters of the recordings and test on the
> quarter the model has never seen, and we repeat that eight different ways so
> the result isn't a fluke."

---

## Section 2 — Results

**What it is:** all 9 algorithms, scored on the same data, same splits.

### The four metrics

Imagine the model is guessing which sign it just saw.

| metric | plain meaning | asks the question |
|---|---|---|
| **Accuracy** | how many it got right overall | "out of 100 recordings, how many correct?" |
| **Precision** | when it says "អរគុណ", how often is that true? | measures **false alarms** |
| **Recall** | of all the real "អរគុណ" recordings, how many did it catch? | measures **misses** |
| **macro-F1** | precision and recall combined into one number | the headline |

**Why precision and recall are separate.** A model can cheat on one. If it almost
never guesses "អរគុណ" but is right the rare times it does, precision looks
excellent while recall is terrible — it's missing nearly all of them. You need
both, which is what F1 combines.

**Why "macro".** Macro means every sign counts equally, regardless of how many
recordings it has. Without it, a model could ignore a rare sign entirely and
still score well. Macro-averaging makes ignoring any sign expensive.

**The ± column** is the standard deviation — how much the score wobbled across
the 8 splits. Small means stable. Large means the result depends on which
recordings happened to be held back.

### The result

LDA scores highest at ~91%. **But the report says several algorithms are tied,
and that's deliberate** — the top few are within one standard deviation of each
other. With only 84 recordings, that gap is smaller than the natural wobble, so
claiming a single winner would be overstating it.

The **real** finding is the gap between the top group (~88–91%) and the bottom
(k-NN at ~47%). That's far too big to be luck.

> **Say this:** "The top three or four are basically tied — we can't separate
> them with this much data. What we *can* say is that the good ones are much
> better than the bad ones, and k-NN is clearly unsuitable."

> **If asked why k-NN is so bad:** k-NN classifies by finding the most similar
> recordings it has seen. With 576 numbers describing each recording, "similar"
> stops being meaningful — everything looks roughly equally far from everything
> else. This is a known problem with distance-based methods on lots of features.

---

## Section 3 — The best algorithm in detail

**What it is:** a closer look at the winner, instead of just one number.

**Per-sign scores (Figure 3).** The average hides which signs are hard. This
chart breaks the score down per sign so you can see where the mistakes are
concentrated. Red bars are the weak ones — those need more recordings.

**The confusion matrix (Figure 4).** This looks intimidating and is actually
simple:

- each **row** = the sign you actually performed
- each **column** = what the computer guessed
- the **diagonal** = correct answers

So a big number on the diagonal is good. A big number off the diagonal tells you
exactly which two signs are being mixed up, which is far more useful than
"85% accurate" — it tells you *what to fix*.

> **Say this:** "The confusion matrix shows which signs get mistaken for each
> other. If two signs look similar to the camera, that's where the errors are,
> and that's where we'd add recordings."

---

## Section 4 — Did the recording conditions matter?

**This is the most interesting section. Take your time on it.**

**What we did.** The 12 recordings of each sign weren't done identically — they
follow a deliberate grid: two lighting levels, two distances from the camera,
three standing positions.

That lets us ask a different question. Instead of hiding a random 25%, we hide
**an entire condition**. For example: train the model *only* on brightly lit
recordings, then test it *only* on dim ones. The model has never seen dim
lighting. Does it still work?

**What we found:**

| change | roughly what it costs |
|---|---|
| standing to one side | small — about 9 points |
| moving nearer or further | small — about 14 points |
| **lighting going dim** | **large** |

So the system copes with you standing in a different spot, but not with the room
being lit differently.

**The twist, and it's a good one to present.** How badly lighting hurts depends
enormously on *which algorithm*, and **not in the order Section 2 would suggest**.
LDA is the most accurate model overall, yet one of the worst when lighting
changes. Bagged Trees is third overall but the best under changed lighting.

**Why, in plain terms:**

- **Tree-based** models (Bagged Trees, Random Forest, Decision Tree) work by
  asking yes/no questions: "was the wrist above this height?" If lighting shifts
  the measurements a little, most of those answers stay the same. They bend.
- **Linear** models (LDA, Logistic Regression, SVM) add up every one of the 576
  numbers with a weight. If lighting nudges all of them together, the total
  shifts and the answer flips. They snap.

> **Say this:** "The most accurate model isn't the most reliable one. If the
> system has to work in a room it hasn't seen, we'd choose Bagged Trees even
> though LDA scores higher on paper."

That's a genuinely good research point, and it's the sort of thing a teacher will
appreciate because it shows you understand that one number isn't the whole story.

**The honest caveat, which is in the report.** We always recorded the bright takes
first and the dim ones afterwards. So "dim lighting" and "recorded later when I
was tired" are tangled together — we can't fully prove the drop is caused by the
light rather than by drifting technique. The fix is easy: have some people record
dim first. **Say this out loud if asked** — admitting it is stronger than hiding
it, and it shows you know what a confound is.

---

## Section 5 — Conclusions

Five bullets summarising the above. Nothing new — it's the "if you read nothing
else" section.

The most important one is the **limitation**: all 84 recordings are from one
person. So the report describes how well the system recognises **that person's**
signing. Whether it works for a stranger is a completely different question and
needs recordings from more of the team.

> **Say this:** "These results are for one signer. We know from our other
> experiments that performance drops a lot on a new person, which is why we're
> collecting recordings from everyone."

---

## Section 6 — Reproducing this

One command regenerates every number, chart and the document itself:

```bash
python algo_comparison/run_var_experiment.py
```

Every algorithm has a fixed random seed, so anyone running it gets **identical**
numbers, not merely similar ones. That's what makes it a real experiment rather
than a one-off demo.

> **Say this:** "It's fully reproducible — one command, and the seeds are fixed,
> so you get exactly the same numbers we did."

---

## Likely questions

**"Why only one person?"**
This task was about testing recording conditions, which one person can do
properly and consistently. Testing across people is the other experiment, and it
needs the whole team's recordings.

**"Why is 84 recordings enough?"**
It's small, and we say so in the limitations. It's enough to separate good
algorithms from bad ones — that gap is large. It's not enough to rank the top
three, which is exactly why we report them as tied instead of picking a winner.

**"Which algorithm should we use?"**
Depends on the situation, which is the point. For a controlled setting, LDA. If
conditions vary, Bagged Trees, because it loses far less when the room changes.

**"Isn't 91% low?"**
It's 7 signs, one signer, 84 recordings, and simple algorithms — no neural
network. Random guessing would be about 14%. The number that matters more is that
it holds up when conditions change.

**"What's macro-F1 again?"**
One number combining "how often it's right when it answers" with "how often it
finds the sign at all", averaged so every sign counts equally.
