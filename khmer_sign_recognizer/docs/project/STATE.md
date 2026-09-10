# Where the project is — 2026-09-07

One page. Read this first when picking the work back up; everything below
points at the file that holds the detail.

---

## The system, in numbers

| | |
|---|---|
| corpora | `khmer` (Task B, freestyle) and `khmer_var` (Task A, 12-cell grid) |
| `khmer` | **7 signers, 1,722 real takes**, 10,332 synthetic at exactly 6:1 |
| `khmer_var` | 4 signers, 337 real takes, 2,022 synthetic |
| signs | 7: ជម្រាប់សួរ អរគុណ ខុស ត្រូវ គ្រួសារ ប៉ា ម៉ាក់ |
| **best model** | **`khmer__tcn` — 93.3 macro-F1 on an unseen signer, and it runs live** |
| worst sign | អរគុណ at 95.5 — every sign is above 95 |

**Always quote the unseen-signer number.** The same features identify *who* is
signing at 98%, so a same-signer score partly measures "can it recognise this
person", which is not the task.

## Use this

```bash
./run_web.sh                    # record, or Recognize with khmer__tcn
python scripts/check_camera.py --seconds 20    # run FIRST if live is bad
```

Every command: [`../guides/COMMANDS.md`](../guides/COMMANDS.md).
Report for the teacher: `algo_comparison/results_khmer_taskA_cap30/Task_A_Report.docx`.

---

## What moved the number, and what did not

**More signers beat everything else.** 2 → 7 signers took unseen-signer from
83.8 to 93.3. For comparison, an entire day of bug-fixing was worth +23.6 on
the other corpus, and no architecture change has been worth more than ~1 point
since.

Fixes that worked (all in PROBLEM_LOG):

| | worth | § |
|---|---|---|
| take-aware splitting in the deep path | the old numbers were meaningless — 100% of val samples had their own take in training | C4 |
| stop inventing missing hand positions | +15.2 unseen-signer | K |
| bone directions (not lengths) | +8.4 unseen-signer | M |
| five more signers | +9.5 unseen-signer | Q |

**Ruled out with measurements — do not retry:**

| | why | § |
|---|---|---|
| more epochs / bigger model | train accuracy is already 100% | R.3 |
| more synthetic | +1 to +2, measured repeatedly | J.3 |
| partial-window training | works (+11.3 at 40%) but improves the wrong metric | R.4 |
| ST-GCN | 46 pair errors vs the TCN's 43 | S |
| a landmark canonicaliser | lost to the cheaper reconstruction | L |
| image enhancement | reached ~57% hand detection, still insufficient | D4 |

---

## The three open threads

**1. ជម្រាប់សួរ ↔ អរគុណ.** The one pair that still confuses. They sit **14.97**
apart where every other pair is 43+ — three times closer. Four modelling
attacks have failed to move it (R.3, R.4, R.1, S), which is strong evidence it
is a *data* problem. **The untried lever is recording more of just those two
signs.** See §5 of the Task A report, which said this before any of it.

**2. Hand tracking.** 92% of the model's signal is the hands, and poor light
drops hand detection to 33–42% (D4). A camera check measured 37.7% / 51.7% in
one session. `check_camera.py` grades it. No model change survives a dim room —
run this before suspecting anything else. (User reports recording in daylight;
worth one confirming run.)

**3. Seng Menghong is an outlier.** Complete 77.9 against 91–99 for everyone
else, and **23 hello/thanks errors** where the next worst is 8. Not the
architecture — ST-GCN and partial training barely moved it. He is also the
signer whose upload held 15 half-recorded takes (Q.3). Points at his recording
session. **Not investigated** — user declined for now.

---

## Things that will bite you

**Run `verify_pool.py` after every import.** A take uploaded with only a clean
view gets no synthetic, which breaks the 6:1 ratio, which makes the take-aware
split group a clip under the wrong parent — the leak. It has no other symptom.

**Never pass `--save` on a smoke-test run.** Twice, a stale or 10-epoch bundle
sat in `models/recognizers/` looking legitimate and was tested for real. When
live behaviour is reported as bad, **check bundle dates before touching code**.

**Synthetic stays out of uploads.** `export_recordings.py` skips it by default.
Shipping it invites a second generation run on top of the first, which is the
mistake behind C2.

**Mid-sign predictions are not answers.** The sliding window measures ~11 points
below the committed answer, and the TCN is the *worst* model on partial input
while being the best on complete. The UI shows tentative guesses as muted
"Reading…" and only the committed answer as "Recognized". Do not restyle them
alike.

---

## Not done

- **The paper draft is stale.** [`PAPER_DRAFT.md`](PAPER_DRAFT.md) argues "deep
  wins on the hard corpus". Seven signers inverted that — architectures
  converge and classical nearly catches deep. *The value of a sequence model
  falls as the corpus grows* is the better and more surprising spine.
- `run_task_a.py` results are not yet in `.icm/experiments/` as registry
  entries.
- Deep architectures have no drop-in folder equivalent to `custom_algos/`; a
  teammate adding one must edit `train.py`.
