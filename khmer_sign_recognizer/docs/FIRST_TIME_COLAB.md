# First-time setup — from zero to trained model on Colab

This guide walks you through the **full v2 workflow** the first time.
Follow it top to bottom. Each step has a **verify** line — don't skip it,
because catching a broken step early saves you an hour later.

Time budget: ~90 minutes the first time. ~10 minutes every time after.

**Prerequisites**: the repo already cloned at `D:\Projects\Sign to Text\khmer_sign_recognizer`,
Windows venv working (camera + MediaPipe + RTMPose already run successfully at least once),
a GitHub account with push access to `Catejsj/Sign-to-Text`,
a Google student account you'll use for Drive.

All commands use **Git Bash** on Windows (comes with Git). Open it via Start → "Git Bash".
If you prefer PowerShell, swap `python` commands the same but path quoting differs.

---

## Step 0 — Open the repo in a terminal

```bash
cd "/d/Projects/Sign to Text/khmer_sign_recognizer"
pwd
```

**Verify**: the `pwd` output ends with `khmer_sign_recognizer`.

---

## Step 1 — Commit and push everything to GitHub

Colab clones from GitHub. Anything uncommitted is invisible to it.

### 1a. See what's pending

```bash
git status --short
```

You should see `M` (modified) and `??` (untracked) lines. That's the work
from the last few weeks that was never pushed.

### 1b. Stage everything that should be tracked

The `.gitignore` already excludes data/, models/, venv/, logs/, `.npy`, `.pth`.
Safe to add everything else:

```bash
git add -A
git status --short
```

Now every line should start with `A`, `M`, `D`, or `R`. No `??`.

### 1c. Commit

```bash
git commit -m "v2 track: Transformer pipeline + Drive sync + cleanup

- add src/v2/ (schema, normalize, augment, dataset, model_transformer, train)
- add scripts/record_motion.py and scripts/drive_sync.py
- add notebooks/colab_train_v2.py
- add WORKFLOW.md
- move ARCHITECTURE.md and SETUP.md into docs/
- delete zlib debug artefacts and day-1 sanity test scripts
- commit previously untracked CNN track files (model, dataset, normalizer, recorder, visualizer, inference, record_signs)"
```

### 1d. Push

```bash
git push origin main
```

The first push from a new machine asks for GitHub credentials.
Use a **personal access token**, not your password:
GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
→ Generate new (scope: `repo`). Paste the token as the password.

**Verify**: open https://github.com/Catejsj/Sign-to-Text in the browser.
You should see `WORKFLOW.md`, `src/v2/`, `scripts/drive_sync.py` in the tree.

---

## Step 2 — Install rclone (Windows)

`rclone` is a one-binary tool that syncs local folders to Google Drive.
We use it so you never have to hand-drag `.npy` files through the browser.

### 2a. Download

Grab `rclone-current-windows-amd64.zip` from https://rclone.org/downloads/

### 2b. Extract to a permanent location

Put `rclone.exe` somewhere stable:

```bash
mkdir -p /c/Tools/rclone
# drag rclone.exe from the downloaded zip into C:\Tools\rclone\
```

### 2c. Add to PATH

PowerShell **as Administrator**:

```powershell
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\Tools\rclone", [EnvironmentVariableTarget]::Machine)
```

**Close and reopen Git Bash** so the new PATH takes effect.

### 2d. Verify

```bash
rclone version
```

**Verify**: prints `rclone v1.xx.x` and OS info. If "command not found",
the PATH didn't take — reopen the terminal, or on this one-off session run
`export PATH="$PATH:/c/Tools/rclone"`.

---

## Step 3 — Configure rclone to point at your student Drive

This is interactive. I'll list every prompt and exactly what to type.

```bash
rclone config
```

Prompts in order:

| Prompt | Type |
|---|---|
| `e/n/d/r/c/s/q>` | `n` (new remote) |
| `name>` | `ksldrive` — exactly this, lowercase |
| `Storage>` | `drive` (or the number matching "Google Drive") |
| `client_id>` | leave blank, press Enter |
| `client_secret>` | leave blank, press Enter |
| `scope>` | `1` (full access — we need it to write) |
| `service_account_file>` | leave blank |
| `Edit advanced config? (y/n)` | `n` |
| `Use web browser to automatically authenticate? (y/n)` | `y` |

A browser tab opens. **Log in with your student Google account**.
Click "Allow". The browser shows "Success!"; the terminal shows your account.

| Prompt | Type |
|---|---|
| `Configure this as a Shared Drive? (y/n)` | `n` |
| `Keep this "ksldrive" remote? (y/e/d)` | `y` |
| final menu | `q` (quit) |

### 3a. Verify

```bash
rclone lsd ksldrive:
```

**Verify**: prints a list of folders in your Drive root (or nothing if it's
empty — that's fine). **No error**.

---

## Step 4 — Create the Drive folder structure

One command:

```bash
rclone mkdir ksldrive:SignLink
rclone mkdir ksldrive:SignLink/data
rclone mkdir ksldrive:SignLink/data/sequences_v2
rclone mkdir ksldrive:SignLink/models
rclone mkdir ksldrive:SignLink/models/weights_v2
rclone mkdir ksldrive:SignLink/logs
rclone mkdir ksldrive:SignLink/logs/v2
```

### 4a. Verify

```bash
python scripts/drive_sync.py doctor
```

**Verify**: lists `data/`, `models/`, `logs/` under `SignLink/`.

---

## Step 5 — Record your first real samples (single streaming session)

We need enough samples to train. Minimum target: **10 signs × 3 signers × 3 takes = 90 samples**.
You can start smaller (you alone, 10 signs × 3 takes = 30 samples) to sanity-check
the pipeline, then expand once it works.

The recorder is a *streaming session* — start it once, do every sign in one
sitting, leave it running. Optionally have WSL + Godot running too so the
mannequin mirrors you while you record.

### 5a. Pick your 10 signs

Write them down. Examples: `hello`, `thanks`, `yes`, `no`, `please`, `sorry`,
`good`, `bad`, `family`, `name`. **Freeze this list** — adding signs later
means retraining from scratch.

### 5b. (Optional but fun) Start the live mannequin

In two separate terminals before the recorder:

```bash
# Terminal 1 — WSL
./start_wsl.sh

# Open khmer-sign-mannequin2 in Godot 4.6, press F5
```

If you skip this, the recording still works — you just won't see the
mannequin. Pass `--no-stream` to the recorder to silence the UDP send.

### 5c. Activate the Windows venv

```bash
source venv/Scripts/activate
python -c "import cv2, mediapipe, rtmlib; print('OK')"
```

**Verify**: prints `OK`. If not, `pip install -r requirements.txt` first.

### 5d. Start the session

```bash
python scripts/record_session.py --signer <your-name>
```

A camera window opens. The terminal shows a blinking cursor (no prompt — type freely).

**The loop, per take**:
1. Type a label, e.g. `hello`, press **ENTER**.
2. The camera overlay shows `GET READY: 1.5s`.
3. The overlay turns red `REC 2.0s left` — perform the sign.
4. The terminal prints `saved [1] hello: alex__real__clean__0000.npy + alex__real__noisy__0000.npy`.
5. Press **ENTER** on an empty line to do another take of `hello`, or type a new label.
6. Repeat for every sign × every take.
7. When done: type `quit` + ENTER, **or** focus the camera window and press `q`.

Stand 1.5–2 m from the camera, arms visible to the waist. If a take captured
fewer than 5 frames the recorder warns and discards it — fix lighting or
hand positioning and redo it.

### 5e. Verify

```bash
ls data/sequences_v2/
ls data/sequences_v2/hello/
```

**Verify**: `data/sequences_v2/` contains one folder per sign.
Each folder has **two files per take** named like:
- `alex__real__clean__0000.npy` (shoulder-normalized — signer-invariant)
- `alex__real__noisy__0000.npy` (raw [0,1] image space — preserves variation)
plus matching `.json` metadata files.

---

## Step 6 — Push data to Drive

```bash
python scripts/drive_sync.py push-data
```

**Verify**: rclone prints a progress bar and ends with
`Transferred: N / N, 100%`. Open
https://drive.google.com/drive/my-drive → `SignLink/data/sequences_v2/`
and confirm the label folders are visible.

---

## Step 7 — Train on Colab

### 7a. Open Colab

Go to https://colab.research.google.com → File → Open notebook → **GitHub** tab.

- Paste the repo URL: `https://github.com/Catejsj/Sign-to-Text`
- Open the file: `khmer_sign_recognizer/notebooks/colab_train_v2.py`

Colab will convert the `.py` into notebook cells automatically
(each `# %%` or `# ==` block becomes a cell).

### 7b. Enable GPU

Runtime → Change runtime type → **T4 GPU** → Save.

### 7c. Run cells one at a time

The file comments say which cell does what. In order:

**Cell 1** — Mount Drive. A popup asks for permission. Click through.

**Cell 2** — Clone/update the repo. On first run it clones; on later
runs it `git reset --hard origin/main` to pick up your latest push.

**Cell 3** — Install deps. `numpy` is already in Colab, torch too;
this is usually a no-op.

**Cell 4** — Sync data Drive → local Colab disk. Should be fast
(just a few MB for 30 samples).

**Cell 5** — Train. You'll see 100 epochs of output like:

```
ep   0  tr_loss 2.3026  tr_acc 0.100  val_loss 2.2980  val_acc 0.111  (4.3s)
ep   1  tr_loss 2.2145  ...
```

With 10 signs and 30 samples, expect:
- **tr_acc** (train accuracy) climbs toward 1.0 within ~20 epochs
- **val_acc** (validation accuracy) is the real number to watch;
  if the dataset is too small, it stays stuck around 0.3–0.5

The "best val acc" at the end is your headline number.

**Cell 6** — Copies the trained `.pt` weights and the log
to Drive with a timestamp. Copy `ksl_transformer_latest.pt` too.

### 7d. Verify

Open Drive → `SignLink/models/weights_v2/`. You should see
`ksl_transformer_latest.pt`, `ksl_transformer_YYYYMMDD_HHMMSS.pt`, and `labels.json`.

---

## Step 8 — Pull weights back to your laptop

```bash
python scripts/drive_sync.py pull-weights
ls models/weights_v2/
```

**Verify**: `ksl_transformer_latest.pt` is in the folder, file size > 0.

---

## What to do if something breaks

| Symptom | Fix |
|---|---|
| `git push` asks for password | Use a **personal access token**, not your password. See step 1d. |
| `rclone: command not found` | PATH didn't update; reopen terminal or run `export PATH="$PATH:/c/Tools/rclone"` |
| `rclone config` browser won't open | Pick `n` at the "auto config" prompt; it gives a URL to paste manually. |
| Colab says `ModuleNotFoundError: No module named 'src'` | You're not inside `khmer_sign_recognizer/`. Cell 2 ends with `%cd /content/Sign-to-Text/khmer_sign_recognizer`. Rerun it. |
| Colab says `no samples found under data/sequences_v2` | Cell 4 didn't run, or your `push-data` didn't actually push. Rerun cell 4 and re-verify Drive. |
| `val_acc` stuck near random (1/10 = 0.1) | You don't have enough data. Record more takes / more signers. |
| `val_acc` very high but model fails on new signers | You trained on one signer. Add teammates as signers, then retrain with `held_out_signer="<one-of-the-signer-names>"` to measure real generalization. |

---

## What happens next week

Once this loop is smooth:

1. **Add signers.** Each teammate runs `record_motion.py --signer <their_name>`.
   This is the single most important thing for accuracy.
2. **Use leave-one-signer-out.** In the Colab `TrainConfig`, set
   `held_out_signer="menghong"` (or whoever). This is the real research measurement.
3. **Godot synthesis pipeline.** Once the real-data baseline is measured,
   we build the headless Godot renderer that generates synthetic samples
   to close signer-generalization gaps. That's a separate setup doc.
