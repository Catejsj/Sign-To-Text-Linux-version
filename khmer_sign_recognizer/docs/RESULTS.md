# Results

What we measured, what it means, and what did not work. Every number here is
read from a `results.json` file, named in each section, so it can be checked
and regenerated with `algo_comparison/run_task_a.py`.

---

## Which run to quote

Two honest runs of the same study exist on the `khmer` corpus:

| run | real takes | best unseen signer | use it for |
|---|---|---|---|
| **`cap30`** | **1,470** | **92.0** (TCN) | anything you write or present |
| `full` | 1,722 | 93.3 (TCN) | the model we ship |

`cap30` caps every signer at their first 30 takes per sign, so nobody
contributes more evidence than anyone else (30 takes × 7 signs × 7 signers =
1,470). Three people recorded more than 30, which is where the other 252 come
from. The extra takes are worth about +0.9, inside the fold noise, so they
belong in the shipped model but not in a protocol claim.

**Never mix the two in one document.** The synthetic result also differs
between them, not just the headline.

Unless a section says otherwise, everything below is
`algo_comparison/results_khmer_taskA_cap30/results.json`.

---

## The dataset

| | `khmer` (recognition corpus) | `khmer_var` (variation grid) |
|---|---|---|
| signs | 7 | 7 |
| signers | 7 | 4 |
| real takes | 1,470 (capped) / 1,722 (all) | 337 |
| recorded | freestyle, any room | 12-cell grid: 2 lighting × 2 distances × 3 positions |
| synthetic | 8,820 (6 per take) | 2,022 |
| used for | headline, model comparison, synthetic experiment | generalisation to held-out conditions |

The seven signs: ជម្រាបសួរ · អរគុណ · ខុស · ត្រូវ · គ្រួសារ · ប៉ា · ម៉ាក់

---

## Headline: a signer the model has never seen

Macro-F1 under **leave-one-signer-out**: train on six people, test on the
seventh, seven times, average.

| model | unseen signer | spread (sd) | same signer | runs live |
|---|---|---|---|---|
| BiLSTM | **92.4** | 5.0 | 98.2 | no |
| **TCN** | **92.0** | 6.5 | 98.3 | **yes, the one we ship** |
| BiGRU | 91.9 | 6.0 | 98.4 | no |
| Gradient Boosting | 91.5 | 7.7 | 97.9 | yes |
| LSTM | 91.3 | 5.2 | 98.3 | yes |
| Transformer | 91.2 | 7.5 | 97.5 | yes |
| GRU | 90.9 | 6.5 | 98.2 | yes |
| Random Forest | 90.0 | 7.1 | 97.5 | yes |
| Logistic Regression | 89.1 | 7.3 | 97.9 | yes |
| Neural Net (MLP) | 88.3 | 7.7 | 97.7 | yes |
| SVM | 86.5 | 8.8 | 97.1 | yes |
| Bagged Trees | 86.5 | 9.0 | 95.4 | yes |
| k-NN | 81.3 | 12.9 | 92.2 | yes |
| LDA | 79.1 | 10.8 | 84.1 | yes |
| Decision Tree | 78.5 | 8.4 | 91.0 | yes |

**Why two columns.** The same features identify *who is signing* at 98.0%
(chance is 14.3%). So a same-signer score partly measures "does it recognise
this person", which is not the task. Report the unseen-signer number, and put
the same-signer one beside it so the gap stays visible.

**Why we ship the TCN and not the BiLSTM.** BiLSTM reads the clip backwards as
well as forwards, so it needs the sign to be finished before it can answer.
Live, there is no future to read. The 0.4-point difference is well inside the
fold-to-fold spread of about ±5.

### Per signer (TCN)

| reaj | Chingsan | Piseth | Vichet | Mao | Mengly | Seng Menghong |
|---|---|---|---|---|---|---|
| 99.0 | 98.1 | 98.1 | 93.4 | 87.3 | 87.1 | **81.0** |

**18 points between the best and worst person, against 1.5 points between all
six neural networks.** Who recorded the data matters more than which model reads
it.

---

## What actually moved the number

| change | effect on unseen signer |
|---|---|
| **2 signers → 7 signers** | 83.8 → 92.0 (93.3 on all takes). The largest single gain |
| bone directions instead of joint positions | median **+9.7** across the 9 classical models |
| stop inventing missing hand positions | median **+4.9** |
| both feature changes together | median **+14.3** |
| any change of architecture | at most ~1 point |

**Bone directions.** Each joint becomes `joint − parent`, divided by its own
length. Direction is kept, length is thrown away, and that is what makes the
representation the same for a tall and a short signer.

**Missing hands.** The old code filled a lost hand with invented coordinates
the model could not tell apart from real ones. Now a gap is filled with the
last known value *and* flagged as a separate feature.

**Temporal order is worth about one point.** The nine classical models
summarise each joint over time as mean, std, min and max, which is
order-invariant: shuffle the frames and the features are identical. They
cannot see the order of the movement at all. Yet the best of them (91.5)
finishes 0.9 behind the best network (92.4), which can.

**Bigger is not better at this scale.** The Transformer has 2.2 million
parameters, thirteen times the GRU's 164 thousand, and beats it by 0.3.

---

## The synthetic mannequin

**Research question:** does training on synthetic 3D mannequin data improve
recognition accuracy compared with real recordings alone?

Every real take is rebuilt on 6 differently proportioned bodies: four scale
factors per body (shoulder width, upper arm, forearm, hand), each drawn from
0.8 to 1.2. Then the 9 classical models are trained twice, real only and real
plus synthetic, with identical features and folds. Both are tested on real
takes only.

| model | real only | + mannequin | change |
|---|---|---|---|
| LDA | 84.07 | 95.25 | +11.18 |
| k-NN | 92.15 | 93.68 | +1.53 |
| SVM | 97.09 | 97.42 | +0.34 |
| Decision Tree | 90.96 | 91.23 | +0.26 |
| Random Forest | 97.48 | 97.62 | +0.14 |
| Logistic Regression | 97.89 | 97.89 | 0.00 |
| Neural Net (MLP) | 97.69 | 97.68 | 0.00 |
| Bagged Trees | 95.45 | 94.42 | −1.03 |
| Gradient Boosting | 97.89 | 96.39 | −1.50 |

**Median +0.14. Five improved, four got worse.**

**Why it did not help.** Normalisation divides out body scale, and the bone
features discard bone length. Stretching bones is most of what the generator
does, so it added examples without adding information. (It does not *only*
change length: scaling a bone also moves the wrist, and hand location carries
meaning in sign language. But what it varies is mostly what our features
ignore.)

**Why LDA jumped.** LDA estimates a covariance matrix over 1,158 features from
1,470 takes, which is badly under-determined. Any extra rows help it, whatever
they contain. It started as the weakest model and finished below where Gradient
Boosting started.

> ⚠️ **This experiment is on the same-signer axis, not leave-one-signer-out.**
> It uses grouped 5-fold cross-validation, where a test fold can contain people
> who also appear in training. The proof: each model's "real only" score above
> equals its same-signer score in the headline table exactly. The split is
> still leak-free (every synthetic copy stays on the same side as its parent),
> but the supported claim is *"did not help under five-fold cross-validation"*,
> not *"did not help on an unseen signer"*. Re-running it under
> leave-one-signer-out is the obvious next experiment.

---

## Generalisation to held-out conditions

From `algo_comparison/results_khmer_var_taskA/results.json`: 337 takes by 4
signers on the 12-cell grid. Each row removes one condition from training
entirely, then tests only on it.

| trained on → tested on | macro-F1 |
|---|---|
| full light → dim light | 93.4 |
| near → far | 94.0 |
| middle and left → right | 91.3 |
| 3 signers → an unseen 4th (BiGRU) | **85.1** |

**Condition shift costs 1 to 3 points. A new person costs nine.**

No condition was reliably *better* either: dim scored 94.6 against 91.7 for
full light, and far 95.2 against 91.1 for near. Both run against intuition and
are within noise on about 168 clips a side. That is why the main corpus was
recorded freestyle and the effort went into recruiting signers instead.

*An older run, `results_khmer_var/results.json`, reports 82.0 / 74.9 / 79.3 for
the same three rows. That was the old feature set and it is superseded.*

---

## Why the numbers can be trusted

| check | result |
|---|---|
| duplicate takes | 0 |
| duplicates across takes | 0 |
| folds disjoint | yes |
| labels shuffled, then everything re-run | 13.4 (chance is 14.3) |
| signer-identity probe on the same features | 98.0 (chance 14.3) |

Every copy of a recording (clean view, noisy view, all six synthetic bodies)
carries its parent's take id, and every split groups on it. This was a real bug
once, which is why it is audited on every run.

**Why cross-validation and not a random split.** We tried 8 random draws, each
holding out a quarter of the takes. 159 of the 1,470 takes were never tested in
any draw, and some were tested six times. Five-fold cross-validation tests every
take exactly once, and it is steadier run to run (sd 1.34 against 1.48).

---

## Where the errors are

| pair | errors |
|---|---|
| **ជម្រាបសួរ ↔ អរគុណ** | **17** (7 one way, 10 the other) |
| ខុស ↔ ម៉ាក់ | 4 |
| ប៉ា ↔ ម៉ាក់ | 3 |

One pair holds most of the confusion. Four modelling attacks on it have failed
(more epochs, partial-window training, ST-GCN, a landmark canonicaliser), which
points at the data. The untried lever is recording more of just those two signs.

Every sign still scores above 95 F1. អរគុណ is lowest at 95.2.

**What the model looks at:** bone directions carry 79.7% of the importance and
raw positions 20.3%. Right hand 53.4%, left hand 39.8%, body only 6.8%.

---

## Live recognition is harder than any number above

**The sliding window costs roughly 9 to 13 points.** Offline numbers score a
complete recording. Live, the system has to answer while you are still signing,
from a partial sign it never trained on.

**The camera is the other half.** In the stored corpus the left hand is missing
in 39.7% of frames and the right in 20.0%; 303 of 1,470 clips never see the left
hand at all. A live check in one room found hands in about 42% of frames while
signing against 92.6% while held still, with the body at 99.7% throughout. That
pattern is **motion blur, not darkness**: the body stays sharp, the fast-moving
hands smear. A software low-light filter only reached about 57% and was removed.
Add real light.

---

## What did not work (do not retry without a new reason)

| | why |
|---|---|
| more epochs or a bigger model | training accuracy is already 100% |
| more synthetic data | +1 to +2 at best, measured repeatedly |
| ST-GCN graph network | 46 errors on the hard pair against the TCN's 43 |
| partial-window training | helps mid-sign (+11.3 at 40%) but optimises the wrong metric |
| a landmark canonicaliser | lost to a cheaper reconstruction |
| image enhancement for low light | reached ~57% hand detection, still not enough; removed |
| rotation augmentation for camera tilt | 94 to 95, against 96.7 flat for de-rolling every frame level |

---

## Limits

- **Seven signs.** A usable system needs hundreds, and the confusable pair
  suggests it gets harder, not easier, as the vocabulary grows.
- **Seven signers, all project members, none fluent Deaf signers.** Nothing here
  is a claim about how Khmer Sign Language is really used.
- **Isolated signs only.** One sign at a time with a clear start and end, not
  continuous signing.
- **No hyperparameter tuning** for any of the fifteen models. The comparison is
  fair, but every number is a floor rather than a ceiling.

## Open threads

1. **Synthetic under leave-one-signer-out.** The experiment that would make the
   research question's answer signer-independent.
2. **Synthetic with joint-position features.** If the mannequin helps there and
   not with bone directions, the explanation becomes a measurement.
3. **Record more ជម្រាបសួរ and អរគុណ.** The data lever on the one pair that
   modelling could not fix.
4. **Multimodal: video alongside landmarks.** Pixels survive exactly where the
   hand tracker fails. The recorder can now capture video and frames alongside
   landmarks; see [USAGE.md](USAGE.md#recording-modes).

---

## The full lab notebook

Code comments cite sections like `PROBLEM_LOG §C2` or `PROBLEM_LOG D4`. That is
the project's running record of every bug, fix and measurement, about 70 KB. It
was retired from the working tree to keep the docs to three files, and lives in
git history:

```bash
git show 47d9630:khmer_sign_recognizer/docs/project/PROBLEM_LOG.md | less
```

Every other retired document (setup notes per OS, the team recording plan, the
paper draft, the term-1 presentation) is recoverable the same way from that
commit.
