# SignLink — Team Recording Plan

Two tasks, both recorded with our normal landmark pipeline. No photos.

| | Task A — teacher task | Task B — recognition pool |
|---|---|---|
| Per sign | **12 takes** on a fixed grid | **30 takes**, sign naturally |
| Per person | 7 × 12 = 84 takes | 7 × 30 = 210 takes |
| Synthetic | 6 per take | 6 per take |
| Drive folder | `TaskA/` | `TaskB/` |

**Everyone records all 7 signs** in both. Not one sign each — if a sign only ever
comes from one person, the model scores well by learning *that person* and
collapses on anyone else.

---

## 1. Recording

Launch as usual:

```bash
./run_web.sh          # Windows: run_web.bat
```

**Record however you like.** The signer tag, the language folder, whatever you
type in the panel — none of it has to be right. The import script fixes all of
it later. Just record.

### Task A — the 12-take grid

Record in the morning. The camera stays straight and still — **you move, not the
camera.**

| take | light | distance | where you stand |
|---|---|---|---|
| 1 | full morning | **near** | middle |
| 2 | full morning | near | left |
| 3 | full morning | near | right |
| 4 | full morning | **far** — about half your body visible | middle |
| 5 | full morning | far | left |
| 6 | full morning | far | right |
| 7 | **dim / half** morning | near | middle |
| 8 | dim / half | near | left |
| 9 | dim / half | near | right |
| 10 | dim / half | **far** | middle |
| 11 | dim / half | far | left |
| 12 | dim / half | far | right |

Takes 7–12 repeat 1–6 exactly, only the light changes.

Left and right still mean **your whole body stays in frame** — step to the edge,
not out of it.

Keep the order, and don't delete a take out of the middle — if you spoil one,
re-record it in its place.

### Task B

All 7 signs, 30 takes each, sign naturally. No conditions.

---

## 2. Uploading

The Drive has exactly two folders. **Make a folder with your name inside the
one you're uploading for, and copy your takes into it.**

```
SignLink/
├── TaskA/
│   ├── sophea/…
│   ├── dara/…
│   └── vann/…
└── TaskB/
    ├── sophea/…
    └── …
```

Easiest way: open the language folder you recorded into —
`khmer_sign_recognizer/data/sequences_v2/khmer_var/` for Task A — select
everything in it, and paste it into your name folder. That's it. The
`sl_001`-style folders come along, which is exactly what's needed.

`python scripts/export_recordings.py` also works if you prefer.

### The only rule

**Keep either the `sl_001`-style folders, or the `.json` files.** Either one on
its own is enough to tell which sign a take is — you only lose data if you strip
*both*, flatten everything into one pile *and* rename the files.

Everything else is handled: any amount of nesting, files named anything, the
signer tag you typed in the panel, whether you included synthetic, and importing
the same person twice.

**Your folder name becomes your signer identity.** That is what makes "can this
recognise someone it has never seen?" answerable — the 96%-vs-63% number in our
results. Two people can both have left the panel's tag as `me` and still
separate correctly, as long as their Drive folders differ.

---

## 3. Pulling it into the project

Download the `TaskA` and `TaskB` folders from the Drive and point the script at
them. **Where you put them doesn't matter** — any path works, and the zip Drive
gives you is accepted as-is, no unzipping:

```bash
python scripts/import_takes.py ~/Downloads/TaskA-20260814T0930Z-001.zip
python scripts/import_takes.py ~/Downloads/TaskB
```

Only the **name** matters, and only loosely — `TaskA`, `task_a`, `TASK A`,
`TaskA (1)` and Drive's `TaskA-20260814T0930Z-001` all work. If you renamed it
to something else, say which task it is:

```bash
python scripts/import_takes.py ~/Downloads/whatever --lang khmer_var
```

That is the whole merge step. Nothing is ever overwritten — incoming takes are
renumbered onto the end of what is already there, so running it again after more
people upload just adds them. Use `--dry-run` first to see who it found:

```
  real takes per person
  person     sl_001   sl_002   sl_003    total
  dara            2        2        2        6
  sophea          2        2        2        6
```

`TaskA` lands in the `khmer_var` language folder and `TaskB` in `khmer`, so the
two never mix. Labels are inherited from `khmer`, so nobody needs a Khmer
keyboard.

### Your own upload coming back to you is fine

You upload your takes, so the pool you download contains them — including the
copy already sitting on your machine. The import compares **take contents**, not
filenames, and skips anything already there:

```
  skipped 84 take(s) already present (same content) — re-import is safe
```

This matters more than it looks. Without it your takes would land a second time
under the name of your Drive folder, so `Piseth` and `piseth` would look like
**two different people** — and holding out one would leave identical copies of
those very takes in the training set. The unseen-signer score would come out
far too high and nothing would look wrong.

So you can import the same pool as many times as you like, and you never need to
delete your local recordings first. (`--allow-duplicates` forces them in, if you
ever actually want that.)

After that, `./run_web.sh` shows both in the language dropdown and training works
normally.

---

## 4. Training

```bash
# regenerate synthetic from the merged real takes
python scripts/generate_synthetic.py --language khmer_var --clean --method scale

# train, holding out one person at a time
python scripts/run_baseline.py --algo rf --lang khmer_var --mode both --holdout <name>
```

**Don't upload synthetic if you can avoid it** — it regenerates from the real
takes in seconds, so it just makes the upload seven times bigger. The import
accepts it if you do.

### Which algorithm to use

See everything available:

```bash
python scripts/run_baseline.py --list
```

Nine are built in: `lda` `svm` `logreg` `knn` `gboost` `mlp` `rf` `nb` `tree`.

### Using your own algorithm

**You do not edit any shared code.** Put a file in `custom_algos/` named
whatever you like — `custom_algos/ridge.py`:

```python
from sklearn.linear_model import RidgeClassifier


def make_ridge():
    return RidgeClassifier(alpha=1.0, random_state=0)


ALGORITHMS = {
    "ridge": ("Ridge Classifier", make_ridge),
}
```

That's it. It now works everywhere:

```bash
python scripts/run_baseline.py --list                 # yours is listed
python scripts/run_baseline.py --algo ridge --lang khmer_var --mode real
python algo_comparison/run_comparison.py --lang khmer_var   # joins the report
```

Because everyone works in their own file, **nobody's algorithm collides with
anyone else's** — no merge conflicts when the folder is shared.

Notes:

- Your factory must **return** the model, not fit it. Pass the function itself
  (`make_ridge`), not a call to it (`make_ridge()`).
- Anything with `.fit(X, y)` / `.predict(X)` works — any sklearn estimator, a
  `Pipeline`, or your own class.
- Features are **already standardized** for everyone, so don't add your own
  scaler — that's what keeps results comparable.
- Pass `random_state=0` wherever it's accepted, or your numbers move between
  runs and can't be compared.
- Reusing a built-in key (e.g. `"rf"`) overrides it, which is the easy way to
  try different parameters. `--list` shows which file it came from.
- A broken file never blocks anyone: the built-ins keep working and the error is
  printed with the filename.

Full guide with examples: `custom_algos/README.md`.

### What to report

Train twice and show both numbers:

- **random split** — will look very high
- **held-out person** (`--holdout`) — the honest number

The gap is the strongest point in the report, and it matches what we already
measured on our own data: 96% same-signer vs 63% unseen signer.

Task A also allows **held-out condition** tests — train on one setting, test on
one never seen. Robustness to something nobody trained on:

| train on | test on |
|---|---|
| takes 1–6 (full light) | takes 7–12 (dim light) |
| near takes (1–3, 7–9) | far takes (4–6, 10–12) |
| middle + left | right |

---

## 5. Optional check

If a merged pool looks wrong, this reports what is actually in it — takes per
person per sign, unpaired views, orphaned synthetic, and the recording
conditions measured from the landmarks:

```bash
python scripts/verify_pool.py --lang khmer_var --expect-takes 12 --conditions
```

---

## 6. For the group chat

> Two tasks, both recorded normally in the panel — no photos, and **don't worry
> about the signer tag or the language folder, they get fixed on import.**
> **Task A** = all 7 signs, **12 takes each**, in the morning. Camera stays
> still, **you move.** Takes **1–3, full morning light, standing near**: middle,
> left, right. Takes **4–6: the same but standing back so only about half your
> body shows** — middle, left, right. Takes **7–12: those same six again in dim
> or half morning light.** Left and right still mean your whole body stays in
> frame. Keep the order and don't delete a take out of the middle. **Task B** = all 7 signs,
> **30 takes**, sign naturally. Everyone does **all 7 signs** in both — if a
> sign comes from only one person the model learns the person, not the sign.
> When you're done run `python scripts/export_recordings.py` and copy what it
> makes into the Drive under **TaskA** or **TaskB**, **inside a folder with your
> name** — that folder name is the only thing that has to be right, it's how we
> tell everyone's recordings apart. One thing not to do: don't strip the
> `sl_001` folders *and* delete the `.json` files *and* rename everything, or we
> can't tell which sign is which. Keeping any one of those three is fine.
