# SignLink — Detailed 2-Year Timeline

The single source of truth for what happens when. Every milestone has a hard deliverable
and a measurable outcome.

---

## Overview

```
            YEAR 2                                       YEAR 3
─────────────────────────────────────────  ────────────────────────────────
TERM 1     TERM 2     TERM 3                T1          T2          T3
───────    ───────    ───────                ───         ───         ───
Foundation Real Data  Final Push             Voice       Vocab+      Deploy
+ Pipeline + LOSO     + Paper                Layer       Continuous  + Paper
                                                          Signing
═══════════════════════════════════════════════════════════════════════════
GOAL:     KSL → Text                       GOAL:    KSL → Speech
DELIVERS: Web app + paper                  DELIVERS: Production app + paper
```

---

## YEAR 2 — KSL → Text

### Term 1 — Foundation (now → next week presentation)

**Goal:** prove the engineering works and the team plan is solid.

| Week | Milestone | Owner | Deliverable |
|---|---|---|---|
| -2 | Pose pipeline working (MediaPipe + RTMPose) | Lead | `src/capture.py`, `record_session.py` |
| -1 | First trained model (smoke test, 1 signer) | Lead | `ksl_transformer_latest.pt` in Drive |
| -1 | Browser VRM mannequin running | Lead | `mannequin_web/index.html` |
| 0 | TCN model added, Transformer retained as baseline | Lead | `src/v2/model_tcn.py` |
| 0 | Architecture documentation | Lead | `docs/SYSTEM_ARCHITECTURE.md`, `docs/PRESENTATION_TERM1.md` |
| **Week of presentation** | **Term 1 review with Dr. May Thu** | All | 15-slide deck, live mannequin demo, smoke-test result |

**Term 1 outcome:** advisors confirm scope, vocabulary (5 signs), and approach.

---

### Term 2 — Real Data + Real Inference (~12 weeks)

**Goal:** first credible cross-signer accuracy result. The system works on people who aren't the lead.

#### Phase 2A — Data collection (weeks 1–3)

| Week | Milestone | Owner | Outcome |
|---|---|---|---|
| 1 | KSL sign validation locked: one reference video per sign | Task 6 owner | `docs/SIGN_REFERENCES.md` |
| 1 | Recording instructions + reference video distributed to team | Lead | All teammates onboarded |
| 1–2 | Each teammate records 3 takes × 5 signs (30 takes per person) | All 8 | ~120 total takes uploaded to Drive |
| 3 | Lead audits all uploads — flag bad takes, request re-recordings | Lead | Clean dataset, ~110 takes (some attrition expected) |

#### Phase 2B — First real training (weeks 3–4)

| Week | Milestone | Owner | Outcome |
|---|---|---|---|
| 3 | First multi-signer training run on full dataset | Lead | TCN trained, weights in Drive |
| 4 | First **real LOSO benchmark** | Lead | LOSO accuracy reported per held-out signer + mean |
| 4 | Confusion matrix + per-class recall logged | Lead | `logs/v2/<run_id>.json` |

**Realistic LOSO target by week 4: 70–80%.** Not 90% yet — that comes after iteration.

#### Phase 2C — Real-time inference (weeks 4–6)

| Week | Milestone | Owner | Outcome |
|---|---|---|---|
| 4 | Real-time inference script started | Task 5 owner | `scripts/infer_live.py` skeleton |
| 5 | 60-frame circular buffer + model load + overlay | Task 5 | Working live demo (camera → predicted text overlay) |
| 6 | Confidence threshold + "uncertain" handling | Task 5 + Task 8 | Demo with confidence display |

#### Phase 2D — Web app skeleton (weeks 6–10)

| Week | Milestone | Owner | Outcome |
|---|---|---|---|
| 6 | FastAPI backend with `/predict` endpoint | Task 4 | Server accepts landmarks, returns predicted sign |
| 7 | React frontend skeleton: webcam + state machine | Task 3 | UI states: ready / recording / predicting / showing |
| 8 | Frontend hooked up to MediaPipe.js + WebSocket/REST | Task 3 + Task 4 | End-to-end browser → server → predicted text |
| 9 | Confidence display + "wrong, it was X" correction button | Task 3 + Task 8 | Correction logging works |
| 10 | Mid-term internal demo | All | Working web demo of recognition |

#### Phase 2E — Iteration (weeks 10–12)

| Week | Milestone | Owner | Outcome |
|---|---|---|---|
| 10 | Identify worst-performing signs/signers from LOSO | Lead | Targeted re-recording list |
| 11 | Re-record problem signs from problem signers | Affected teammates | Cleaner dataset |
| 12 | Retrain, second LOSO benchmark | Lead | Hopefully 80–85% mean LOSO |

**Term 2 outcome:** working web app demo + measurable cross-signer accuracy.

---

### Term 3 — Final Push (~12 weeks)

**Goal:** ship the Year-2 deliverable. ≥90% LOSO + paper draft.

#### Phase 3A — Push to ≥90% (weeks 1–4)

| Week | Milestone | Owner | Outcome |
|---|---|---|---|
| 1 | Hyperparameter tuning (LR, dropout, augmentation strength) | Lead | Tuning sweep results |
| 2 | Augmentation ablation: paired clean/noisy on vs off | Lead | Ablation table for paper |
| 3 | Architecture comparison: TCN vs Transformer vs BiGRU | Lead | Final model + 2 baselines |
| 4 | If LOSO < 90%: add 1–2 takes per sign per signer; retrain | All | Boosted dataset |

#### Phase 3B — Paper draft (weeks 4–8)

| Week | Section | Owner |
|---|---|---|
| 4 | Introduction + Related Work | Lead + Task 5 owner |
| 5 | Methodology (architecture + pipeline) | Lead |
| 6 | Dataset description + Methodology | Lead + Task 6 owner |
| 7 | Results (LOSO + ablations + confusion matrix) | Lead |
| 8 | Discussion + Limitations + Future work | Lead |

#### Phase 3C — Final polish (weeks 8–11)

| Week | Milestone | Owner | Outcome |
|---|---|---|---|
| 8 | Confidence + correction loop working | Task 8 owner | Production-quality UX |
| 9 | Sign-to-text output layer (sentence builder, optional TTS hint) | Task 7 owner | Polished demo output |
| 10 | Mannequin polish (if time) | Task 2 owner | Visual demo improvements |
| 11 | Demo rehearsal + paper review | All | Ready for final |

#### Phase 3D — Final presentation (week 12)

| Milestone | Deliverable |
|---|---|
| Final presentation to Dr. May Thu | 25-min talk, live demo, paper draft |
| Open-source repo cleanup | Final README, archived experiments |
| Term 3 report submission | Year-2 deliverable bundle |

**Year-2 outcome to Dr. May:**
1. Working web app: webcam → KSL sign predicted as text
2. ≥90% LOSO accuracy on 5 signs
3. Full research paper draft
4. Open-source repo with documentation
5. Live mannequin demo

---

## YEAR 3 — KSL → Speech

### Term 1 — Voice Layer

| Week | Milestone | Outcome |
|---|---|---|
| 1–2 | Khmer text-to-speech library evaluation | Pick a TTS engine (Coqui, eSpeak-NG, or a Khmer-specific solution) |
| 3–4 | TTS integration into web app | Predicted sign → spoken Khmer audio |
| 5–6 | Pilot user testing with deaf community contacts | Real UX feedback |
| 7–10 | Iteration on output design (sentence pacing, voice choice) | Production-grade output layer |
| 11–12 | Term 1 paper section: speech synthesis layer | Paper extension draft |

### Term 2 — Vocabulary Expansion + Continuous Signing

| Week | Milestone | Outcome |
|---|---|---|
| 1–4 | Expand vocabulary from 5 → 20–30 signs | New reference videos, retraining |
| 5–8 | Continuous signing: detect sign boundaries in a sentence | Real research challenge — model needs temporal segmentation |
| 9–10 | Sentence-level evaluation (BLEU or word error rate) | Quantitative continuous signing result |
| 11–12 | Year 3 mid-term review | Continuous demo + benchmark |

### Term 3 — Deployment + Paper Submission

| Week | Milestone | Outcome |
|---|---|---|
| 1–4 | Web app polish + deployment to public URL | Anyone can try it |
| 5–8 | Paper writing: full Year 3 paper combining recognition + speech + continuous signing | Complete paper |
| 9–10 | Submit paper to a venue (regional AI conference, ACM SIGACCESS, or IEEE) | Submitted |
| 11–12 | Final defence / graduation requirement | Year 3 review |

---

## Critical Path & Dependencies

```
KSL sign validation (Task 6)
        │
        ▼
Multi-signer recording  ◄────  Recording instructions locked
        │
        ▼
Multi-signer training run
        │
        ├──▶  Real-time inference script (Task 5)
        │
        └──▶  Web app backend (Task 4)
                       │
                       ▼
              Web app frontend (Task 3)
                       │
                       ▼
                  Year-2 demo
```

**Anything that delays KSL sign validation delays everything else.** Make sure that task ships in Term 2 week 1.

---

## Single-Person Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Lead burns out from carrying solo | High | Catastrophic | Delegate Task 6 + Task 5 + a web app task before Term 2 starts |
| Teammates don't deliver Tasks 3, 4, 5 | Medium | High | Lead has skeleton implementations of each in `scripts/` so the lights stay on |
| Dataset quality is bad (wrong signs) | Medium | High | Task 6 first; Lead audits all uploads in Term 2 week 3 |
| Colab GPU quotas / infra issues | Low | Medium | Free tier is enough for our scale; can always pay $10/month if blocked |
| Mannequin requirement scope-creeps | Medium | Medium | Lock the mannequin as visualisation-only, no synthetic data in Year 2 |

---

## How to Use This Timeline

- **Lead:** review this every Sunday evening. Mark the current week. Anything not on track → identify *which task owner* is blocking and message them directly.
- **Team:** each owner reads only their own track. Don't try to absorb the whole 2-year plan at once.
- **Mentor (Dr. May Thu):** the Year-2 column is what we report on. Year 3 is provisional.

This document changes. When it changes, commit it with a message like `timeline: shift Task 5 to week 5 (data delays)`. Future-you will thank present-you for the audit trail.
