# SignLink — Quick Start

SignLink recognises isolated **Khmer Sign Language (KSL)** signs from a webcam.
This guide takes you from a fresh download to recording training data.

> The project runs **fully local** — one webcam, one process. No WSL, no
> Godot, no browser. If you followed older docs that mention `run_windows.py`,
> WSL, or a Godot mannequin — those were removed. This guide is current.

---

## 1. Get the code

Download the ZIP from <https://github.com/Catejsj/Sign-to-Text> — green
**Code** button → **Download ZIP** — and unzip it. (Or `git clone` it.)

Everything below runs from the `khmer_sign_recognizer` folder.

---

## 2. One-time setup

Requires **Python 3.12** and an **NVIDIA GPU**. Open PowerShell:

```powershell
cd path\to\Sign-to-Text\khmer_sign_recognizer
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Every time you open a new terminal, `cd` in and re-run
`.\venv\Scripts\Activate.ps1`.

---

## 3. Record training data — this is the main task

```powershell
python scripts\record_session.py --signer yourname
```

Use the **same short lowercase tag** every time you record (e.g. `piseth`).
It goes into every filename so we can test the model on a held-out signer.

A camera window opens. In the terminal:

- type a **sign label** + ENTER → 1.5s countdown → 2s recording → saved
- press ENTER on an **empty line** → another take of the same sign
- type `quit` (or press `q` in the camera window) → end the session

Record these **10 signs, 3 takes minimum each**. Do not change the list:

```
hello   thanks   yes   no   please   sorry   good   bad   family   name
```

Saved takes land in `data/sequences_v2/<label>/`.

---

## 4. See the 3D mannequin (optional)

A 3D humanoid that mirrors your movement, driven by the same tracking:

```powershell
python scripts\mannequin_local.py
```

Keys (focus the camera window): `q` quit · `m` show/hide the mannequin.

---

## 5. Send your data to the team

After a recording session:

```powershell
python scripts\drive_sync.py push-data
```

(rclone must be configured first — ask the lead if `drive_sync.py` errors.)

---

## 6. Training (Google Colab — the lead runs this)

Open `notebooks/colab_train_v2.py` in Google Colab and run the cells in order.
Trained weights land back in the shared Drive.

---

## Hard rules

1. **Same signer tag** every time you record.
2. **Code in Git, data + weights in Drive** — never commit `.npy` or `.pt`.
3. **Don't change the 10-sign list** without team agreement — a new sign
   means retraining from scratch.

---

## Script reference

| Script | What it does |
|---|---|
| `scripts/record_session.py` | record training data (RTMPose body + MediaPipe hands) |
| `scripts/mannequin_local.py` | live 3D mannequin viewer |
| `scripts/compare_detectors.py` | diagnostic: RTMPose vs MediaPipe accuracy/jitter |
| `scripts/drive_sync.py` | sync data and weights with Google Drive |
