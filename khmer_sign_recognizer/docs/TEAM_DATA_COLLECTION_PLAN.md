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
camera.** Stand far enough back that about half your body is visible and **stay
at that distance the whole time.**

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

Easiest way to get your files:

```bash
python scripts/export_recordings.py
```

Then copy what it produced into your folder. Copying the raw
`data/sequences_v2/…` folders straight out of the project works too.

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

Download the `TaskA` and `TaskB` folders from the Drive (the browser's "download
folder" is fine — it arrives as a zip, so unzip it first), then:

```bash
python scripts/import_takes.py ~/Downloads/TaskA
python scripts/import_takes.py ~/Downloads/TaskB
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

### What to report

Train twice and show both numbers:

- **random split** — will look very high
- **held-out person** (`--holdout`) — the honest number

The gap is the strongest point in the report, and it matches what we already
measured on our own data: 96% same-signer vs 63% unseen signer.

Task A also allows a **held-out condition** test — train on takes 1–6 (full
light), test on 7–12 (dim). Robustness to something nobody trained on.

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
> still, **you move.** Stand back so about half your body shows and stay at that
> distance. Takes **1–6 in full morning light**: middle, left, right, then
> middle, left, right again. Takes **7–12: the same six in dim light.** Keep the
> order and don't delete a take out of the middle. **Task B** = all 7 signs,
> **30 takes**, sign naturally. Everyone does **all 7 signs** in both — if a
> sign comes from only one person the model learns the person, not the sign.
> When you're done run `python scripts/export_recordings.py` and copy what it
> makes into the Drive under **TaskA** or **TaskB**, **inside a folder with your
> name** — that folder name is the only thing that has to be right, it's how we
> tell everyone's recordings apart. One thing not to do: don't strip the
> `sl_001` folders *and* delete the `.json` files *and* rename everything, or we
> can't tell which sign is which. Keeping any one of those three is fine.
