# Synthetic Data — Explaining The Second Experiment

A companion to `Task_A_Synthetic_Comparison.docx`. Same style as
`TASK_A_EXPLAINED.md`: what we did, what it means, and what to say.

Everything here is about **khmer_var only** — the 12-takes-per-sign grid
recording from 4 people. No other dataset is involved.

---

## The one-paragraph version

> We had 337 real recordings. That's not many, so we generated 6 extra training
> examples from each one by mathematically stretching the recorded skeleton to
> different body sizes — same movement, different body. We then trained every
> algorithm twice: once on the real recordings only, once on real plus
> synthetic. We always tested on real recordings. The result was a **small,
> mixed improvement** — about 2 points on average, with one algorithm getting
> dramatically worse.

---

## 1. What synthetic data actually is here

**It is not a fake video and not an AI-generated image.** Nothing is drawn or
imagined.

Each real recording is a list of joint positions. To make a synthetic version we
take that list and **change the bone lengths** — longer arms, shorter forearms,
narrower shoulders — while keeping every joint *angle* exactly as recorded.

The result is the same gesture performed by a differently-proportioned body.

**Why the sign cannot change.** A sign is defined by what the arms do — the
angles and the movement. Scaling a bone's length does not rotate it, so the
angles are mathematically untouched. We measured this: with deliberately extreme
stretching, the largest change in any elbow angle across a whole clip was
**0.0004°** — floating-point noise.

> **Say this:** "We stretch the skeleton to different body sizes but never change
> the joint angles, so the sign is provably identical. It's geometry, not
> generation — no camera, no AI drawing anything."

> **If asked how it differs from AI image generation:** an image generator might
> produce something that no longer means the same sign, and you'd have no way to
> prove otherwise. Here the movement is preserved by construction, so we can
> guarantee the label is still correct.

---

## 2. Why we bothered

We have **12 recordings per sign per person**. That is very little. Most of the
algorithms were clearly struggling — k-NN managed only 45%.

Recording more takes costs time from four people. Generating synthetic variants
costs seconds and no one has to be in the room. If it works, it's free data.

**It turned out to help far less than we first thought**, and the reason that
matters is in Section 6 — our first attempt at this measurement was wrong, and
finding out why is arguably the most useful thing in this experiment.

---

## 3. How we made the test honest

This is the part to emphasise, because "we added data and the score went up" is
worthless without it.

**Rule 1 — we only ever test on real recordings.**
If we scored the model on synthetic data we'd be measuring whether it can
recognise our own arithmetic, not whether it can recognise signs.

**Rule 2 — a recording's synthetic copies stay on its side of the split.**
This is the one that's easy to get wrong. Every real take has 6 synthetic
children. If a take goes into the test set, all 6 of its children must go into
the test set too — never into training. Otherwise the model trains on
near-copies of the very recording it's about to be tested on, and the score is
meaningless.

**We verified this directly, and the first time we checked it we were not
checking hard enough** — see Section 6. The test that matters is not "does each
group contain one real recording" but "is each synthetic copy grouped with the
recording it actually came from". Because stretching a skeleton leaves the joint
angles untouched, we can match every synthetic copy back to its true parent by
comparing angles. All 72 copies we tested matched the parent they were grouped
with.

> **Say this:** "We never test on synthetic data, and a recording's synthetic
> copies always travel with it — if the original is in the test set, so are all
> its copies. We checked that directly rather than trusting the code."

> **This is worth volunteering before you're asked.** It's the first thing a
> sceptical reader should wonder about, and having the answer ready is the
> difference between a result and a claim.

---

## 4. What we found

**A small, mixed improvement.** Five of nine algorithms improved, three got
worse, one barely moved. The median change is **+1.9 points**.

That is a real effect but a modest one — not the transformation you might expect
from multiplying the training data by six.

> **Say this:** "Synthetic data gave us about two points on average. It helped
> the algorithms that were short of examples and did nothing for the ones that
> weren't. It's worth having, but it isn't a substitute for recording more."

### The one result that needs explaining: LDA

LDA fell **30 points**, far more than anything else moved in either direction.
Don't average that away — explain it, because it is genuinely informative.

LDA works by estimating how all 576 features vary together within each sign,
which means inverting a large covariance matrix. Synthetic variants of the same
recording are very similar to each other — and the view we train on already
divides out overall body size, which is most of what the stretching changes. So
adding six near-copies of every recording multiplies the sample count without
adding much genuinely new variation, and that makes the covariance estimate
unstable.

Algorithms that don't invert a covariance matrix — trees, k-NN, logistic
regression — are untroubled by it.

> **Say this:** "LDA got much worse, and that's a real effect, not a bug. It
> needs to estimate how features vary together, and adding lots of near-identical
> copies makes that estimate unstable. It's a good illustration that more data
> isn't automatically better for every method."

### The algorithms did not converge

The spread between best and worst was 37 points on real data and 38 with
synthetic. **The choice of algorithm still matters just as much.** Synthetic data
did not level the field.

## 5. The honest limits

**This does not replace recording more people.** Every synthetic example is
derived from an existing real recording, so it can only vary body proportions —
not how a *different person* actually signs, their timing, their habits. The
report's other experiment (holding out a whole person) is where that shows up.

**The gain has a ceiling.** On our larger corpus — 30 recordings per sign instead
of 12 — the same test showed synthetic making things very slightly *worse*. Once
the model has enough real data, extra warped copies add nothing and cost a little
precision.

> **Say this if asked "so should we always use synthetic data?"** — "No. It gains
> a lot when you're short of recordings and nothing once you have enough. On our
> denser dataset it slightly hurt. It buys back the recordings you didn't make,
> up to a point."

**All four people appear in both training and testing.** So these figures
describe recognition for people the system has seen. A stranger is a separate
question.

---

## 6. We measured this wrong the first time

Worth telling, because it is the most useful thing that happened.

**The first run said synthetic data was transformative** — every algorithm
improved by 13 to 47 points, k-NN going from 45% to 93%. That would have been a
headline result. It was wrong.

**What went wrong.** The generator had been run **twice** by mistake, so each
recording had 12 synthetic copies instead of 6. The code that keeps a recording's
copies together assumed the copies were numbered in one continuous run per
parent. With two runs the numbering interleaved: copies 0–5 belonged to
recording 0, copies 6–11 to recording 1 — but the grouping put all twelve with
recording 0.

The consequence: when recording 1 was held out for testing, **its own copies
stayed in the training set**. The model was tested on a recording whose warped
twins it had just been trained on. Scores rose accordingly.

**How it surfaced.** Not from the code — from regenerating the data cleanly for a
different reason (to get back to the agreed 6 copies). The scores dropped by 20
to 40 points, and that difference is what exposed the problem.

**How we confirmed it.** Stretching a skeleton leaves the joint angles unchanged,
so a synthetic copy can be matched back to its true parent by comparing angles.
After regenerating, all 72 copies tested matched the parent they were grouped
with. Before, the numbering made that impossible.

> **Say this:** "Our first measurement was too good. It turned out the synthetic
> copies weren't being kept with the right original, so the model was being
> tested on warped versions of recordings it had trained on. We found it by
> regenerating the data and seeing the scores fall by 30 points, and we
> confirmed it by matching each copy back to its true original using the fact
> that stretching doesn't change joint angles."

**The lesson worth stating.** A leak makes results *better*, never worse, so
there is nothing to alert you. Our original check — "does every group contain
exactly one real recording?" — passed, and it was the wrong question. The right
one was "is each copy grouped with the recording it actually came from?"

**A result that looks too good deserves the same scrutiny as one that looks
wrong.**

---

## 7. If the teacher asks…

**"Isn't this just duplicating your data?"**
No — a duplicate would be an identical copy, which teaches nothing. Each
synthetic version has different body proportions, so the model sees the same sign
performed by a different build. That's what stops it from memorising one body
shape.

**"How do you know it's not cheating?"**
We never test on synthetic data, and a recording's copies always stay on the same
side of the split as the original. We verified that by matching every copy back
to its true parent through joint angles, not just by trusting the code — and as
Section 6 explains, we only started checking that carefully after an earlier
version of this experiment gave results that were too good.

**"Why 6 synthetic copies per recording?"**
It's a setting we chose. More copies means more training data but also more
repetition of the same underlying recording, so there's a point past which extra
copies stop adding variety.

**"Could you just record more instead?"**
Yes, and that would be better — real recordings capture real variation that
geometry can't invent. Synthetic data is what you do when recording more isn't
practical, and our results say it recovers a large part of the gap.

**"Which result should we believe — with or without?"**
Both are real; they answer different questions. Without synthetic tells you how
much the recordings alone support. With synthetic tells you how well the system
can be made to work today. For deciding which algorithm to deploy, use the
synthetic numbers. For deciding whether to record more, look at how big the gain
was — a large gain means the dataset is too small.
