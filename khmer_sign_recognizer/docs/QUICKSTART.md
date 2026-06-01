# SignLink — Quick Start

SignLink is a **Khmer Sign Language (KSL) recognition** project. We record
short skeleton clips of signs from a webcam, then a model learns to
classify them. This guide gets you from zero to recording.

> **The project changed a lot recently.** If you followed older docs that
> mention `run_windows.py`, WSL, Godot, a tk label window, or
> `mannequin_web/` — **all of that is gone**. This doc is the current one.
> Anything else under `docs/` is older context, not setup instructions.

---

## 1. What you need

- **Python 3.12** (Windows; macOS/Linux probably work but untested)
- **An NVIDIA GPU** (RTMPose body tracking runs on CUDA)
- **A webcam**
- About **2 GB free disk** for the Python deps

---

## 2. Get the code

Option A — Download ZIP:

1. <https://github.com/Catejsj/Sign-to-Text>
2. Green **Code** button → **Download ZIP**
3. Unzip somewhere — every command below runs inside the
   `khmer_sign_recognizer` folder of what you unzipped

Option B — Clone:

```powershell
git clone https://github.com/Catejsj/Sign-to-Text.git
cd Sign-to-Text\khmer_sign_recognizer
```

---

## 3. One-time setup

Open PowerShell **in the `khmer_sign_recognizer` folder** and run:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

That installs OpenCV, MediaPipe, RTMPose (`rtmlib`), Open3D, Pillow, and
PyTorch with CUDA support.

**If PowerShell refuses to run `Activate.ps1`**, run this once (admin
PowerShell):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Every new terminal:** `cd` into the folder and re-run
`.\venv\Scripts\Activate.ps1` before any `python` command.

---

## 4. Drop in the Khmer font (one-time, important)

So Khmer labels display correctly on the camera overlay:

1. Download **Noto Sans Khmer** from
   <https://fonts.google.com/noto/specimen/Noto+Sans+Khmer>
2. Unzip; find `NotoSansKhmer-Regular.ttf` in the `static/` folder
3. Copy it into `khmer_sign_recognizer\fonts\`
   - Final path: `khmer_sign_recognizer\fonts\NotoSansKhmer-Regular.ttf`

If you skip this, Khmer text on the camera shows as boxes. Browser
typing still works either way — browsers handle Unicode themselves.

---

## 5. Record training data (the main task)

```powershell
python scripts\record_session.py --signer yourname
```

Replace `yourname` with **your short lowercase tag** (e.g. `piseth`,
`menghong`). Use the **same tag every time you record** — it goes into
every filename so we can validate on a held-out signer.

### What opens

Three windows / panels:

1. **Terminal** — status logs (saved take confirmations)
2. **Camera window** — your live feed with the tracked skeleton overlay
3. **Browser tab** (opens automatically) — the **label input UI**
4. **Open3D window** — a 3D synthetic mannequin posed live by your motion
   (visual demo of synthetic generation)

### How to record a take

In the **browser tab**:

1. Type a sign label (Khmer, English, ASL gloss — any language) and press
   **ENTER**.
2. Camera shows **COUNTDOWN** (1.5 s) — get ready.
3. Camera shows **RECORDING** (up to 3 s) — perform the sign.
4. Press **SPACE** in the camera window or click **Stop** in the browser
   to end early. Otherwise it auto-stops at 3 s (hard cap 8 s).
5. The take is saved. Status returns to **IDLE**.

### Recording multiple takes of the SAME sign

After your first take, the sign shows in the **Recently used** list in
the browser. **Click it to record another take** — no retyping.

If you click while a take is still recording, the next take **queues**
automatically. You'll see a "next take queued: X" banner. The new take
auto-starts when the current one finishes.

### The 10 KSL signs

```
សួស្តី     អរគុណ     ជម្រាប់សួរ     ទេ     សូម
សុំទោស     ល្អ        អាក្រក់         គ្រួសារ  ឈ្មោះ
```

**3 takes minimum per sign per signer**. More is better.

### End the session

Either:
- Press **`q`** in the camera window, OR
- Click **Quit** in the browser tab

---

## 6. The useful flags

You almost never need them, but:

```powershell
# Generate synthetic body-variants on every saved take (recommended for the lead):
python scripts\record_session.py --signer piseth --synthetic 6

# Record in a different language (default is khmer):
python scripts\record_session.py --signer piseth --lang english

# Disable the 3D mannequin window if it's slowing your PC:
python scripts\record_session.py --signer piseth --mannequin 0

# Set a longer ceiling on per-take recording (default 3 seconds):
python scripts\record_session.py --signer piseth --duration 5
```

`--synthetic 6` means each real take is augmented into 6 synthetic
signers with different body proportions (arm length, shoulder width,
hand size). Same motion, different body. Saved as extra `.npy` files
tagged `synthetic`.

---

## 7. Where takes get saved

```
data/sequences_v2/<lang>/<slug>/
├── piseth__real__clean__0000.npy        ← (60, 48, 3) shoulder-normalized
├── piseth__real__clean__0000.json       ← metadata
├── piseth__real__noisy__0000.npy        ← (60, 48, 3) raw [0,1] coords
└── piseth__real__noisy__0000.json
```

Plus, per language:
```
data/sequences_v2/<lang>/labels.json     ← {"sl_001": "សួស្តី", ...}
```

`labels.json` maps the ASCII slug used in filenames back to the original
Unicode label you typed. ASCII slug for filesystems, Unicode for humans.

With `--synthetic 6`, each take adds 6 × 2 × 2 = **24 extra files**
named `..._synthetic_...`.

---

## 8. Send your data to the team's Google Drive

```powershell
python scripts\drive_sync.py push-data
```

This uses **rclone**, which must be configured first. If
`drive_sync.py` errors out, ping the lead — they'll set up the rclone
remote called `ksldrive` for your machine.

To pull trained weights back from Drive (after the lead has trained):
```powershell
python scripts\drive_sync.py pull-weights
```

---

## 9. Training (the lead does this, on Colab)

Open `notebooks/colab_train_v2.py` in Google Colab and run the cells.
Weights land back in the shared Drive.

---

## Hard rules

1. **Always use the same signer tag** when you record — leave-one-signer-out validation depends on it.
2. **Code in Git, data and weights in Drive** — never commit `.npy` or `.pt` files.
3. **Don't change the 10 signs** without team agreement — adding a new sign means retraining from scratch.

---

## Other scripts (mostly diagnostic, you won't need them)

| Script | What it does |
|---|---|
| `scripts/record_session.py` | **main** — record training data |
| `scripts/mannequin_local.py` | standalone 3D mannequin viewer (no recording) |
| `scripts/compare_detectors.py` | diagnostic: RTMPose vs MediaPipe accuracy/jitter |
| `scripts/generate_synthetic.py` | batch-generate synthetic from existing recordings (rarely needed — `--synthetic` flag covers most cases) |
| `scripts/drive_sync.py` | sync data and weights with Google Drive |

---

## Troubleshooting

**`Activate.ps1 cannot be loaded ...`** — run the `Set-ExecutionPolicy` line under §3.

**`MediaPipe Tasks init failed` / `CUDA not available`** — your venv didn't get GPU PyTorch. Re-run the `pip install torch --index-url ...cu121` line.

**Khmer renders as `?????` in the camera overlay** — drop
`NotoSansKhmer-Regular.ttf` into `fonts/` (see §4).

**Khmer renders as `?????` when I type in the browser** — your Windows
Khmer keyboard layout isn't set up. Settings → Time & Language → Khmer →
Keyboard → NIDA Khmer Unicode. Or copy-paste from elsewhere. Or click
labels from the "Recently used" list and never type them.

**The Open3D window is slow on my PC** — pass `--mannequin 0` to disable it.
