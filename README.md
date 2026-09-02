# SignLink — Khmer Sign Language recognition

Recognises isolated Khmer Sign Language signs from an ordinary webcam and turns
them into text.

A person signs at their camera. The system extracts their skeleton — 48 joints
across the body, wrists and both hands — normalises the movement so that one
signer's body proportions look like anyone else's, and classifies the result.
No motion-capture hardware, no gloves, no depth camera.

```
webcam ──▶ MediaPipe + RTMPose ──▶ 48 joints × 60 frames ──▶ classifier ──▶ text
                                    (2 seconds of movement)
```

> **About the repo name.** This started as the Linux rebuild of an earlier
> Windows-only project. The code is now cross-platform — Linux, Windows and
> macOS all run the same pipeline, and only the install steps differ.

---

## Does it work?

Yes, with an honest caveat, and the caveat is the interesting part.

Two corpora of the same 7 signs, both recorded with the landmark pipeline:

| | signs | takes / sign | real takes | signers |
|---|---|---|---|---|
| **`khmer`** — recognition corpus | 7 | 30 | 420 | 2 |
| **`khmer_var`** — variation grid | 7 | 12 | 337 | 4 |

**On the recognition corpus, tested on the person who recorded it:** 97%
macro-F1 (logistic regression). **Tested on a signer the model has never
seen,** the same setup gets 44%.

| algorithm | same signer | unseen signer | drop |
|---|---|---|---|
| Logistic Regression | 97.1 | 43.6 | −53.5 |
| Random Forest | 96.9 | 59.3 | −37.6 |
| **Bagged Trees** | **96.0** | **75.0** | **−21.0** |
| Gradient Boosting | 95.0 | 36.2 | −58.8 |
| SVM | 94.0 | 27.1 | −66.9 |

*Mean macro-F1 (%), 5 take-aware splits; unseen-signer column is
leave-one-signer-out over both signers.*

**The ranking inverts.** The algorithm that wins on your own data is not the
one that survives a new person — the best same-signer model loses the most,
and the model ranked third generalises best by a wide margin. Any number
quoted without saying which signer it was tested on is close to meaningless.

Full write-up: [`docs/results/`](khmer_sign_recognizer/docs/results/).

---

## What we learned from the variation grid

`khmer_var` was recorded deliberately: every sign performed 12 times across
2 lighting conditions × 2 distances × 3 standing positions, by 4 people in
4 different rooms. Then we trained on one condition and tested on another.

| trained → tested | macro-F1 |
|---|---|
| bright → dim | **82.0** |
| middle/left → right | 79.3 |
| near → far | **74.9** |

*Real takes only. `algo_comparison/results_khmer_var/results.json`.*

With a *single* signer, changing the lighting was catastrophic. With four
people in four rooms it costs a couple of points. **Recruiting more signers
fixed the problem that tighter lighting control was supposed to fix** — and
camera distance, not light, is now the expensive variable.

### Synthetic data helps, a little

We generate extra training signers by rebuilding each recorded take on
differently-proportioned bodies — the same motion on a different skeleton,
pure geometry rather than a generative model. Joint angles are provably
unchanged by the transform, so the sign survives it
([how it works](khmer_sign_recognizer/docs/reference/SYNTHETIC_RETARGETING.md)).

Across 9 algorithms: 5 improve, 3 worsen, 1 flat — **median +1.9 points.**
The gain is small because the normalisation already divides out body scale,
which is most of what the retargeting varies. LDA is the one casualty,
collapsing 71.0 → 40.8: near-duplicate rows destabilise the covariance matrix
it has to invert.

---

## Getting started

Everything lives in [`khmer_sign_recognizer/`](khmer_sign_recognizer/).

```bash
git clone https://github.com/Catejsj/Sign-To-Text-Linux-version.git
cd Sign-To-Text-Linux-version/khmer_sign_recognizer
```

Then follow the setup guide for your machine:

- **Linux** → [`docs/setup/LINUX_REBUILD.md`](khmer_sign_recognizer/docs/setup/LINUX_REBUILD.md)
- **Windows** → [`docs/setup/SETUP_WINDOWS.md`](khmer_sign_recognizer/docs/setup/SETUP_WINDOWS.md)
- **macOS** → [`docs/setup/SETUP_MACOS.md`](khmer_sign_recognizer/docs/setup/SETUP_MACOS.md)

Once installed, one command gets you recording:

```bash
./run_web.sh
```

Joining the team? Start at
[`docs/guides/TEAM_ONBOARDING.md`](khmer_sign_recognizer/docs/guides/TEAM_ONBOARDING.md).
Every other document is indexed in
[`docs/README.md`](khmer_sign_recognizer/docs/README.md).

---

## Repository layout

```
Sign-To-Text-Linux-version/
└── khmer_sign_recognizer/          the project
    ├── run_web.sh                  ← start here: the control panel
    ├── src/                        capture, normalisation, models
    ├── scripts/                    recording, import, synthetic, training
    ├── webapp/                     the control panel (record + recognize)
    ├── algo_comparison/            experiment drivers and reports
    ├── custom_algos/               drop a file here to add an algorithm
    ├── signlang_image_lab/         separate image-based side experiment
    └── docs/                       setup, guides, reference, results
```

Recorded takes, trained weights and the virtual environment are **not** in
git — they are large and machine-specific, and travel by Drive instead. The
shared `labels.json` files are the exception, so everyone's recorder agrees on
what each sign is called.

---

## Data and privacy

Recordings are stored as **skeleton coordinates only** — arrays of joint
positions. No video or images of anyone are kept, committed, or uploaded, and
a landmark file cannot be turned back into a picture of the person who
recorded it.
