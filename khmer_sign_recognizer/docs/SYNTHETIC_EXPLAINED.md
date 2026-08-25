# Synthetic Data — Explaining The Second Experiment

A companion to `Task_A_Synthetic_Comparison.docx`. Same style as
`TASK_A_EXPLAINED.md`: what we did, what it means, and what to say.

Everything here is about **khmer_var only** — the 12-takes-per-sign grid
recording from 4 people. No other dataset is involved.

---

## The one-paragraph version

> We had 337 real recordings. That's not many, so we generated extra training
> examples by mathematically stretching each recorded skeleton to different body
> sizes — same movement, different body. We then trained every algorithm twice:
> once on the real recordings only, once on real plus synthetic. We always
> tested on real recordings. Adding synthetic data improved every single
> algorithm, and it helped the weak ones most.

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

---

## 3. How we made the test honest

This is the part to emphasise, because "we added data and the score went up" is
worthless without it.

**Rule 1 — we only ever test on real recordings.**
If we scored the model on synthetic data we'd be measuring whether it can
recognise our own arithmetic, not whether it can recognise signs.

**Rule 2 — a recording's synthetic copies stay on its side of the split.**
This is the one that's easy to get wrong. Every real take has 12 synthetic
children. If a take goes into the test set, all 12 of its children must go into
the test set too — never into training. Otherwise the model trains on
near-copies of the very recording it's about to be tested on, and the score is
meaningless.

**We verified this directly**, rather than assuming it: the 337 real takes form
337 groups, each holding exactly one real recording plus its own children, and
no group appears on both sides of any split.

> **Say this:** "We never test on synthetic data, and a recording's synthetic
> copies always travel with it — if the original is in the test set, so are all
> its copies. We checked that directly rather than trusting the code."

> **This is worth volunteering before you're asked.** It's the first thing a
> sceptical reader should wonder about, and having the answer ready is the
> difference between a result and a claim.

---

## 4. What we found

**Every algorithm improved.** The report has the exact numbers.

**The interesting part is who improved most.** The algorithms that were *worst*
on real data alone gained the most; the ones already doing well gained least.

That pattern tells you *why* it worked. Synthetic data doesn't teach the model
anything new about Khmer signs — every synthetic example comes from a real one.
What it does is give the model **more examples to learn the same patterns from**.
Algorithms that were failing because they didn't have enough data benefit
enormously. Algorithms that had already extracted what there was to extract
benefit barely at all.

> **Say this:** "Synthetic data doesn't add information — it adds examples. So
> the algorithms that were starved gained the most, and the ones already doing
> well gained the least. That's exactly the pattern you'd expect if our real
> problem is simply not having enough recordings."

### Why k-NN in particular explodes upward

k-NN classifies a new recording by finding the most similar recordings it has
already seen. With only ~250 training examples spread across 7 signs and 576
features, its nearest neighbours were often not similar at all. Multiply the
training set by twelve and its neighbours become genuinely close. It's the
algorithm most dependent on data density, so it gains the most.

---

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

## 6. If the teacher asks…

**"Isn't this just duplicating your data?"**
No — a duplicate would be an identical copy, which teaches nothing. Each
synthetic version has different body proportions, so the model sees the same sign
performed by a different build. That's what stops it from memorising one body
shape.

**"How do you know it's not cheating?"**
We never test on synthetic data, and a recording's synthetic copies always stay
on the same side of the split as the original. We verified that directly, and
reproduced the whole result with a second, independently written piece of code.

**"Why 12 synthetic copies per recording?"**
It's a setting. More copies means more training data but also more repetition of
the same underlying recording; there's a point past which extra copies stop
adding variety. 12 is what was generated for this dataset.

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
