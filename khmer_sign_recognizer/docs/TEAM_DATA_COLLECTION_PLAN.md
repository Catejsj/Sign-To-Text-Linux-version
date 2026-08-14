# SignLink — Team Data Collection Plan

Guideline for all 8 members. **Both tasks are landmark recordings** made with our
own pipeline — no photos. The teacher said "images" but gave us freedom, so we
record the way we always do and treat her requirement as a *sample-count and
variation* requirement instead.

Because both tasks use the same pipeline, the same 7 signs and the same signer
tags, **nothing in a filename tells them apart.** Section 1 fixes that.

| | Task A — teacher task | Task B — recognition pool |
|---|---|---|
| Constraint | **fixed 12-take grid** (light × position) | freestyle, our normal protocol |
| Per person | 7 signs × 12 takes = **84 takes** | 7 signs × 30 takes = **210 takes** |
| Synthetic | 6 per take | 6 per take |
| When | **morning** | any time |
| Language folder | **`khmer_var`** | **`khmer`** |
| Purpose | show robustness to light and position | best possible recogniser |

The 7 signs and slugs are identical in both. Never invent new ones.

| slug | sign |
|---|---|
| `sl_001` | ជម្រាប់សួរ |
| `sl_002` | អរគុណ |
| `sl_003` | ខុស |
| `sl_004` | ត្រូវ |
| `sl_005` | គ្រួសារ |
| `sl_006` | ប៉ា |
| `sl_007` | ម៉ាក់ |

---

## 1. How the two tasks stay separated

**Use the language folder.** It is a real directory on disk, so the two corpora
physically cannot mix, and every tool already understands it:

```
data/sequences_v2/
├── khmer/          ← Task B  (already holds Piseth + Vichet)
│   ├── labels.json
│   └── sl_001/ … sl_007/
└── khmer_var/      ← Task A  (new)
    ├── labels.json     ← identical copy of khmer's
    └── sl_001/ … sl_007/
```

- In the web panel, pick the language from the dropdown **before recording** —
  the same control you already use. Create `khmer_var` once with the "new
  folder" option.
- To train: `--lang khmer` or `--lang khmer_var`.
- `expA` and `expB` in your data are the same trick already used for an earlier
  experiment, so this is a proven pattern, not a new one.

**Why not a tag or a note instead:**

- Changing your signer tag per task (`dara_a` / `dara_b`) would make one human
  look like two signers and destroy leave-one-signer-out — your own data would
  land in train *and* test.
- `SampleMeta.notes` exists in the schema but **nothing reads or filters on it**,
  and it never appears in the filename. It cannot separate anything today.

**They can be merged later, but never un-merged.** Keep them apart now; if we
decide the variation data helps the main recogniser, copying it in afterwards is
safe because signer tags are unique.

> `khmer_var` is recorded with today's pipeline, so its `clean` view is
> de-rolled and matches `khmer`. The old imported folders (`autsl*`, `expA/B`)
> are **not** de-rolled — never mix those into a Khmer training run.

---

## 2. Before anything — claim your signer tag

Every filename starts with your tag. Get this wrong and the pool corrupts
silently.

1. **Lowercase, ASCII, no spaces** — `sopheak`, `menghong`, `dara`.
2. **One tag per person, forever, identical in both tasks.** Typing `sok` today
   and `Sok` tomorrow makes the system think you are two people — your own data
   then appears in both training and test, and the cross-signer score becomes a
   lie.
3. **No duplicates.** Two people named Dara → `dara_l` and `dara_s`.
4. **Never leave the default `me`.** The panel refuses an empty tag but accepts
   `me`, and every member would then write `me__real__clean__0000.npy`, which
   overwrite each other on merge.

Claim it in `00_SIGNER_TAGS.md` on the Drive **before recording.**

> Existing data uses `Piseth` and `Vichet` (capitalised). Leave those alone —
> the filenames and their `.json` sidecars must agree. New members use
> lowercase.

---

## 3. Why everyone records **all 7 signs**

The obvious split — "you do sign 1, I do sign 2" — **breaks the experiment.**

If `sl_001` only ever comes from Sopheak, the model can score 100% by learning
*Sopheak's hands, sleeves, room and webcam* and never the sign. Class identity
becomes perfectly confounded with person identity: the accuracy looks excellent,
means nothing, and collapses on the first stranger.

So the design is **crossed, not blocked** — every person contributes every sign,
in both tasks. Then the only thing separating `sl_001` from `sl_002` is the sign.

Same lesson as `docs/SYNTHETIC_RETARGETING.md` §7: *the evaluation must vary the
factor you claim to be robust to.*

---

## 4. Task A — the teacher task (variation constrained)

### 4.1 The 12-take grid

**12 takes per sign, 6 synthetic generated per take.** Record in the morning.

The camera stays straight and still the whole time — **you move, not the
camera.** Same sign every take.

**Distance stays the same for all 12 takes** — stand far enough back that about
half your body is visible. Do not move closer or further at any point.

| take | light | where you stand |
|---|---|---|
| 1 | full morning | middle |
| 2 | full morning | left |
| 3 | full morning | right |
| 4 | full morning | middle |
| 5 | full morning | left |
| 6 | full morning | right |
| 7 | **dim / half** morning | middle |
| 8 | dim / half | left |
| 9 | dim / half | right |
| 10 | dim / half | middle |
| 11 | dim / half | left |
| 12 | dim / half | right |

Takes 4–6 repeat 1–3, and 7–12 repeat the whole set in dim light. Two takes of
every combination.

Left and right still mean **your whole body stays visible in frame** — step to
the edge, not out of it.

**Order is fixed.** Take 1 = variant `0000` … take 12 = variant `0011`, so the
condition is recoverable from the variant number with no extra metadata. Do not
delete a take out of the middle — if you spoil one, delete it and re-record it
in place.

### 4.2 We verify the conditions, we do not trust them

Even with exact setups, eight people in eight rooms will drift. So the condition
is **measured from the recording afterwards**, not taken on faith — the `noisy`
view already contains everything needed:

| what changed | measured from the landmarks |
|---|---|
| middle / left / right | **body centre offset** from frame centre |
| full vs dim light | **detection dropout rate** — frames where the hands were not found |
| distance (should *not* change) | **shoulder width** — flags anyone who drifted closer |

`verify_pool.py` prints these per person for each of the 12 slots.

This also means the report can state the variation **as a number** ("body centre
spans 0.28–0.71 of frame width across positions") rather than as a claim that
people were told to stand in different places. That is the difference between a
described protocol and a measured one.

### 4.3 Sample count

Per person: 7 signs × 12 takes = **84 takes.**

Pooled per sign folder:

| | 8 people | if one drops out (7) |
|---|---|---|
| real takes | 8 × 12 = **96** | 84 |
| + 6 synthetic each | **672** | **588** |

That clears the teacher's "at least 500 per sign folder" **and survives someone
not finishing** — which is why it is 12 and not 10 (10 would give 490 with a
dropout, just under the line). We report the 96 real separately from the 672
total, honestly.

> If she means 500 *real* takes per sign, that is 63 per sign per person — 441
> each, five times this plan. Ask her before anyone records; do not silently
> record more.

### 4.4 What to report

Train each model **twice** and show both numbers:

- **random take split** — will look very high
- **held-out person split** (`--holdout <tag>`) — the honest number

The gap between them is the strongest point in the whole report, and it matches
what we already measured: 96% same-signer vs 63% unseen signer.

Task A also gives two **held-out condition** tests — train on one setting, test
on the one never seen:

| train on | test on |
|---|---|
| takes 1–6 (full light) | takes 7–12 (dim light) |
| middle + left | right |

Direct measurements of robustness to something nobody trained on — the result
the teacher actually asked for.

---

## 5. Task B — recognition pool (freestyle)

Our normal protocol, into `khmer`.

- All 7 signs, **30 real takes each** = 210 takes.
- Sign naturally. No condition schedule.
- Piseth and Vichet are already done — their 210 takes each are in `khmer` now.
  With 8 signers, leave-one-signer-out becomes 8-fold and the unseen-signer
  number finally becomes trustworthy.

---

## 6. Recording, exporting, uploading

### 6.1 Record

```bash
./run_web.sh          # Windows: run_web.bat
```

In the panel, **before the first take**: set your signer tag, then pick the
language folder (`khmer_var` for Task A, `khmer` for Task B). Check both are
right on screen — the overlay shows the signer tag.

### 6.2 Do **not** upload synthetic

Set synthetic to 0 while recording for the pool, or just do not export it.

1. Synthetic regenerates from the real `.npy` files in seconds — uploading it
   multiplies transfer size ~7× for zero information.
2. Deleting a real take **orphans its synthetic children**; the parent link
   becomes underivable and that data is silently dropped from training
   (`docs/SYNTHETIC_RETARGETING.md` §8).

Generate it once, locally, after the pool is merged. `export_recordings.py`
already skips synthetic by default — do not pass `--include-synthetic`.

### 6.3 Export

```bash
python scripts/export_recordings.py --signer <your_tag>
```

Produces `exports/<your_tag>/` mirroring the data layout, **including both
language folders**. Drag that one folder to the Drive.

Both the `.npy` **and** its `.json` sidecar must travel. A take missing its
sidecar is unreadable.

### 6.4 Merging is a plain copy — why it is safe

Variant numbers are allocated per `(label, signer, source, view)` *within a
language folder*. Two different signer tags therefore cannot produce the same
filename, so merging never overwrites anything. Copy each person's
`sequences_v2/` over `data/sequences_v2/` — nothing is renamed or renumbered.

The only ways to break it: a duplicated tag, an inconsistent tag, a missing
`.json`, a different `labels.json`, or recording into the wrong language folder.

### 6.5 `labels.json` is canonical — download it, never write it

It maps slug → Khmer text. If one person's copy has a different sign at
`sl_003`, their takes land in the wrong class and the error is invisible.

The authoritative copy lives on the Drive. **Download it and overwrite your
local file, in both `khmer/` and `khmer_var/`, before recording.** Do not add,
reorder or rename entries.

This also solves the missing Khmer keyboard — nobody ever types Khmer. You
record into slugs and the Khmer text comes from the shared file.

### 6.6 After merging

```bash
# 1. verify the pool (exits non-zero if anything is wrong)
python scripts/verify_pool.py --lang khmer_var --expect-takes 12 --conditions

# 2. regenerate synthetic, per language folder
python scripts/generate_synthetic.py --language khmer     --clean --method scale
python scripts/generate_synthetic.py --language khmer_var --clean --method scale

# 3. train, holding out one signer at a time
python scripts/run_baseline.py --algo rf --lang khmer_var --mode both --holdout <tag>
```

---

## 7. Drive structure

```
SignLink/
├── 00_SIGNER_TAGS.md                  ← claim your tag here first
├── labels.json                        ← CANONICAL, download before recording
├── _GUIDE.md                          ← this file
├── task_a_variation/
│   └── <tag>/sequences_v2/khmer_var/sl_00X/…
├── task_b_recognition/
│   └── <tag>/sequences_v2/khmer/sl_00X/…
└── results/
    └── <algo>.json                    ← one per person
```

Upload only into **your own** `<tag>` folder. Never edit anyone else's, and
never edit `labels.json`.

---

## 8. Tooling

**Built:**

- **`scripts/init_language.py`** — clone `labels.json` into a new language
  folder so nobody has to retype Khmer. Also detects a label mismatch against
  the canonical copy, which is the one error that silently puts takes in the
  wrong class.

  ```bash
  python scripts/init_language.py --from khmer --to khmer_var
  ```

- **`scripts/verify_pool.py`** — run after merging, before training. Catches
  duplicate or inconsistently-cased signer tags, missing `.json` sidecars,
  unpaired clean/noisy views, bad shapes, NaN/inf, label folders not in
  `labels.json`, orphaned synthetic, and anyone missing a sign. Exits non-zero
  so it can gate a training run. `--conditions` additionally measures body
  centre, shoulder width and hand-detection loss per grid slot.

  ```bash
  python scripts/verify_pool.py --lang khmer_var --expect-takes 12 --conditions
  ```

**Still to build:**

- **Condition-holdout support** — a `--holdout-condition` option on
  `run_baseline.py` so the train-on-full-light/test-on-dim result can be
  produced directly instead of by hand.

---

## 9. One-paragraph version for the group chat

> Two tasks, both recorded with our normal landmark pipeline — **no photos.**
> They are kept apart by the **language folder** you pick in the panel, so check
> it before every session. **Task A (teacher)** → folder `khmer_var`: all 7
> signs, **12 takes each, 6 synthetic per take**, recorded **in the morning**.
> The camera stays straight and still — **you move, not the camera.** Stand far
> enough back that about half your body is visible and **stay at that distance
> the whole time.** Takes **1–6, full morning light:** middle, left, right, then
> middle, left, right again. Takes **7–12: the same six in dim or half morning
> light.** Left and right still mean your whole body stays in frame. Keep the
> order — take 1 is variant 0000, take 12 is 0011 — and don't delete a take out
> of the middle; re-record it in place.
> **Task B** → folder `khmer`: all 7 signs, **30 takes**, sign naturally. Everyone records **all 7 signs** in both — if a sign comes
> from only one person the model learns the person, not the sign. Claim a
> lowercase signer tag in `00_SIGNER_TAGS.md` first and never change it. When
> done: `python scripts/export_recordings.py --signer <your_tag>` and upload the
> one folder it makes. **Do not upload synthetic** — we regenerate it after
> merging. Download `labels.json` from the Drive before you record.
