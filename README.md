
# SignLink — Khmer Sign Language recognition

Recognises isolated Khmer Sign Language signs from an ordinary webcam and turns
them into text.

A person signs at their camera. The system extracts their skeleton — 48 joints
across the body, wrists and both hands — normalises the movement so that one
signer's body proportions look like anyone else's, and classifies the result.
No motion-capture hardware, no gloves, no depth camera.

```
webcam ──▶ MediaPipe + RTMPose ──▶ 48 joints × 30 frames ──▶ classifier ──▶ text
                                    (2 seconds of movement)
```

> **About the repo name.** This started as the Linux rebuild of an earlier
> Windows-only project. The code is now cross-platform — Linux, Windows and
> macOS all run the same pipeline, and only the install steps differ.

---

## 👋 Scanned the QR code on our poster?

**Start here.** This is the code, the data and the full write-up behind
*From Gesture to Text*, presented at Cambodia University of Technology and
Science.

https://github.com/user-attachments/assets/749b0e66-d853-40d9-8376-217905f24fff

|  |  |
|---|---|
| **92.0%** | correct on a signer the system has **never seen** |
| **98.3%** | on a signer it has seen before |
| **15** | models compared under one protocol |
| **7** | signers · 7 signs · 1,722 recordings |

Every score is **leave-one-signer-out**: train on six people, test on the
seventh, seven times over. So the headline number describes a stranger walking
up to the camera, not one of us.

**The question we set out to answer — and the answer we got:**

> *Does training on a synthetic 3D mannequin improve accuracy compared with
> real recordings alone?*
>
> **No. Median gain: one tenth of a point.** Five algorithms improved, four got
> worse, and the one large gain belongs to the weakest model in the study. The
> reason is in our own feature design: we describe a joint by the direction it
> points and discard bone length — and length is the one thing the mannequin
> varies.

**What it cannot do yet.** Every number here is measured offline. Live
recognition is harder — the system decides mid-sign from a sliding window,
before the sign has finished, which costs roughly ten points. Seven signs is a
demonstration, not a vocabulary. And everyone who recorded is a project member
rather than a fluent Deaf signer, so nothing here is a claim about Khmer Sign
Language as the Deaf community actually uses it.

> **Why the poster says 92.0 and the table below says 93.3.** Two honest runs
> of the same study. The poster uses `cap30`, which caps every signer at the
> first 30 takes per sign so that nobody contributes more evidence than anyone
> else — 1,470 of the 1,722 recordings. The table below uses all 1,722. The
> capped run is the stricter protocol and the one in our written report, so it
> is what we print.

---

## Does it work?

Yes, with an honest caveat, and the caveat is the interesting part.

Two corpora of the same 7 signs, both recorded with the landmark pipeline:

| | signs | takes / sign | real takes | signers |
|---|---|---|---|---|
| **`khmer`** — recognition corpus | 7 | 30+ | **1,722** | **7** |
| **`khmer_var`** — variation grid | 7 | 12 | 337 | 4 |

**Tested on a signer the model has never seen** — the number that matters,
because that is what happens when a stranger uses it:

| model | same signer | unseen signer | runs live? |
|---|---|---|---|
| **TCN** | 98.1 | **93.3** | yes |
| BiLSTM | 98.5 | 93.1 | no — reads the clip backwards |
| GRU | 98.4 | 92.4 | yes |
| Gradient Boosting | 98.0 | 91.9 | yes (classical) |
| Random Forest | 98.0 | 90.6 | yes (classical) |

*Macro-F1, leave-one-signer-out over all 7 signers. Every sign scores above 95.*

**Why both columns.** The same features that recognise signs identify *who is
signing* at 98%, so a same-signer score partly measures "can it recognise this
person" — which is not the task. Any number quoted without saying which signer
it was tested on is close to meaningless.

**What moved this number was people, not models.** With 2 signers the best
unseen-signer score was 83.8; with 7 it is 93.3. Over the same period the gap
between the best and worst architecture shrank to about 1 point — as the corpus
grew, the choice of model stopped mattering and the choice of *who recorded*
started to.

Full write-up: [`docs/RESULTS.md`](khmer_sign_recognizer/docs/RESULTS.md).

---

## What we learned from the variation grid

`khmer_var` was recorded deliberately: every sign performed 12 times across
2 lighting conditions × 2 distances × 3 standing positions, by 4 people in
4 different rooms. Then we removed one condition from training entirely and
tested only on it.

| trained → tested | macro-F1 |
|---|---|
| bright → dim | 93.4 |
| near → far | 94.0 |
| middle/left → right | 91.3 |
| 3 signers → an unseen 4th | **85.1** |

*`algo_comparison/results_khmer_var_taskA/results.json`, current feature set.*

**Condition shift costs 1 to 3 points. A new person costs nine.** No condition
was reliably better either: dim beat bright and far beat near, both within
noise. **Recruiting more signers fixed the problem that tighter condition
control was supposed to fix**, which is why the main corpus was recorded
freestyle.

*An older run with the previous feature set reported 82.0 / 74.9 / 79.3 for the
first three rows, and concluded that distance was the expensive variable. Under
the current features near→far is the best of the three, so that conclusion no
longer holds.*

### Synthetic data does not help

We generate extra training signers by rebuilding each recorded take on
differently-proportioned bodies — the same motion on a different skeleton,
pure geometry rather than a generative model. Joint angles are provably
unchanged by the transform, so the sign survives it
([details](khmer_sign_recognizer/docs/RESULTS.md#the-synthetic-mannequin)).

On the current 7-signer corpus it does **not** help. Across the same 9
algorithms: 5 improve, 4 worsen — **median +0.1 points**, and the one large
gain (+11.2) belongs to LDA, the weakest model in the study, which still
finishes below almost everything else.

The reason is in the representation, not the generator. Normalisation already
divides out body scale, and the bone-direction features deliberately discard
bone length — which is the only thing the retargeting varies. It adds examples
rather than information.

*An earlier version of this README reported median +1.9 from a smaller corpus
and an older feature set. The figure above is from
`algo_comparison/results_khmer_taskA_cap30/results.json` and is the one used on
the poster and in the written report.*

*One caveat on that experiment: it was scored with five-fold cross-validation,
where a test fold can contain people also seen in training, not with
leave-one-signer-out like the headline. The split is leak-free, but the claim
it supports is "did not help under cross-validation". Re-running it per signer
is the next experiment.*

---

## Getting started

Everything lives in [`khmer_sign_recognizer/`](khmer_sign_recognizer/).

```bash
git clone https://github.com/Catejsj/Sign-To-Text-Linux-version.git
cd Sign-To-Text-Linux-version/khmer_sign_recognizer
```

There are three documents, and that is all of them:

| | |
|---|---|
| [`docs/SETUP.md`](khmer_sign_recognizer/docs/SETUP.md) | install on Linux, Windows or macOS |
| [`docs/USAGE.md`](khmer_sign_recognizer/docs/USAGE.md) | record, share, train, evaluate, recognize |
| [`docs/RESULTS.md`](khmer_sign_recognizer/docs/RESULTS.md) | every number, what it means, and what did not work |

Once installed, one command gets you recording:

```bash
./run_web.sh
```

---

## Repository layout

```
Sign-To-Text-Linux-version/
└── khmer_sign_recognizer/          the project
    ├── run_web.sh                  ← start here: the control panel
    ├── webapp/                     the control panel (record + recognize)
    ├── src/                        capture, normalisation, features, models
    ├── scripts/                    record, check, share, import, synthetic, train
    ├── algo_comparison/            the 15-model comparison and its report
    ├── custom_algos/               drop a file here to add an algorithm
    └── docs/                       setup, usage, results
```

Recorded takes, trained weights and the virtual environment are **not** in
git — they are large and machine-specific, and travel by Drive instead. The
shared `labels.json` files are the exception, so everyone's recorder agrees on
what each sign is called.

---

## Data and privacy

By default, recordings are stored as **skeleton coordinates only** — arrays of
joint positions. A landmark file cannot be turned back into a picture of the
person who recorded it, and every result above comes from landmark data.

The recorder can also save **video and still frames** for multimodal
experiments, but only when switched on per session. Those files contain faces,
stay on the recording machine under `data/media/`, and are never committed to
git. Anyone recorded on video needs to have agreed to video specifically.
