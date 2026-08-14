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

Export what you recorded:

```bash
python scripts/export_recordings.py
```

Then drag the folder it makes into the Drive, **inside a folder with your name**:

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

**The folder name is the only thing that matters.** It becomes your signer
identity on import, which is what makes "can this recognise someone it has never
seen?" answerable. Everything else — filenames, the tag you typed, how deeply
nested it is — is sorted out automatically.

Two people can use the same name inside the panel and it still works, as long as
their Drive folders differ.

---

## 3. Importing

Download `TaskA/` and `TaskB/` and run:

```bash
python scripts/import_takes.py ~/Downloads/TaskA
python scripts/import_takes.py ~/Downloads/TaskB
```

That is the whole merge step. It handles any folder nesting, missing or broken
`.json` sidecars, files named anything, real and synthetic, and repeated imports
of the same person — nothing is ever overwritten, incoming takes are renumbered
onto the end of what is already there. Add `--dry-run` to look first.

`TaskA` lands in the `khmer_var` language folder and `TaskB` in `khmer`, so the
two never mix. Labels are inherited from `khmer`, so nobody needs a Khmer
keyboard.

After that, `./run_web.sh` shows both folders in the language dropdown and
training works normally.

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
> When you're done run `python scripts/export_recordings.py` and drop the folder
> into the Drive under **TaskA** or **TaskB**, inside a folder with your name.
> The folder name is the only thing that has to be right.
