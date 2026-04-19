# Khmer Sign Recognizer — Architecture & Changes Documentation

> **Last updated:** April 14, 2026
> **Project location:** `D:\Projects\Sign to Text\khmer_sign_recognizer`
> **Python:** 3.12.8 (Windows) / System python3 (WSL)

---

## Table of Contents

1. [How the System Works — The Big Picture](#1-how-the-system-works)
2. [What Changed — Before vs After](#2-what-changed)
3. [Complete File Map — What Every File Does](#3-file-map)
4. [The Two Pipelines Explained](#4-the-two-pipelines)
5. [How to Run Everything](#5-how-to-run-everything)
6. [How to Add New Features Without Breaking Old Ones](#6-adding-new-features)
7. [What Goes in Git vs What Stays Local](#7-git-rules)
8. [Dependency Management](#8-dependencies)
9. [Known Issues & Future Work](#9-known-issues)

---

## 1. How the System Works

The project has **two independent pipelines** that share the same camera capture code but serve different purposes:

```
┌─────────────────────────────────────────────────────────────────┐
│                     SHARED FOUNDATION                           │
│                                                                 │
│   Camera → RTMPose (body, GPU) + MediaPipe (hands, CPU)         │
│                    src/capture.py                               │
│              "Dual Pipeline Capture"                            │
└────────────────┬──────────────────────┬─────────────────────────┘
                 │                      │
     ┌───────────▼──────────┐  ┌────────▼──────────────────────┐
     │ PIPELINE A: Live Demo│  │ PIPELINE B: AI Training       │
     │                      │  │                               │
     │ run_windows.py       │  │ record_signs.py               │
     │      ↓               │  │      ↓                        │
     │ send_to_wsl.py (UDP) │  │ src/normalizer.py             │
     │      ↓               │  │ src/recorder.py               │
     │ WSL: main_wsl.py     │  │      ↓                        │
     │ src/mapper.py        │  │ Saves .npy files              │
     │ src/bridge.py (UDP)  │  │ data/sequences/{label}/       │
     │      ↓               │  │      ↓                        │
     │ Godot 3D Mannequin   │  │ Verify with visualizer:       │
     │ (Y-Bot, pretty)      │  │ python -m src.visualizer      │
     │                      │  │      ↓                        │
     │ PURPOSE:             │  │ Train in Colab (later)        │
     │ Show it working      │  │      ↓                        │
     │ Presentations/demos  │  │ inference.py                  │
     └──────────────────────┘  │ Live sign → text prediction   │
                               │                               │
                               │ PURPOSE:                      │
                               │ Build the actual AI            │
                               └───────────────────────────────┘
```

**Key insight:** Pipeline A and Pipeline B never interfere with each other. They share `capture.py` but everything downstream is separate. You can run one without the other.

---

## 2. What Changed — Before vs After

### The project was moved from `E:\` (SSD) to `D:\` drive

| What | Before | After |
|------|--------|-------|
| Project path | `E:\Projects\Projects\Sign to Text\...` | `D:\Projects\Sign to Text\...` |
| Windows venv | Broken (hardcoded E: paths) | Fresh `venv\` on D: |
| WSL venv | Broken (hardcoded E: paths) | Fresh `venv_wsl\` on D: |
| `start_wsl.sh` path | `/mnt/e/Projects/Projects/...` | `/mnt/d/Projects/...` |
| `pyvenv.cfg` | Pointed to E: | Auto-generated for D: |

### Requirements were updated

| Package | Was | Now | Why |
|---------|-----|-----|-----|
| `opencv-python` | 4.10.0.84 | 4.13.0.92 | Latest stable |
| `mediapipe` | 0.10.14 | **0.10.14 (kept)** | 0.10.18+ removed `mp.solutions.holistic` which we need |
| `rtmlib` | unpinned | 0.0.15 | Pinned for reproducibility |
| `onnxruntime` | unpinned, CPU-only | `onnxruntime-gpu` | RTMPose was running on CPU — now uses GPU |
| `numpy` | 1.26.4 | 2.4.4 | Latest (fallback to 1.26.4 if issues) |
| `python-json-logger` | 2.0.7 | 4.1.0 | Latest |
| `torch` | not installed | added | Needed for model inference |

> ⚠️ **mediapipe is locked to 0.10.14.** Do NOT upgrade it. Google removed the `solutions.holistic` API in 0.10.18, and our `capture.py` depends on it.

### New files added (AI training pipeline)

| File | Purpose | Touches old code? |
|------|---------|-------------------|
| `src/normalizer.py` | Shoulder-center normalization (Windows-only, no WSL) | No — standalone port of mapper logic |
| `src/recorder.py` | Buffers frames → saves .npy with clean + raw versions | No — new module |
| `src/model.py` | Temporal CNN architecture (PyTorch) | No — new module |
| `src/dataset.py` | PyTorch Dataset that reads .npy folder structure | No — new module |
| `src/visualizer.py` | 2D stick figure replay for recorded .npy files | No — new module |
| `record_signs.py` | Recording UI — type label, SPACE to record | No — new top-level script |
| `inference.py` | Live sign recognition with trained model | No — new top-level script |
| `notebooks/train_ksl.py` | Self-contained Colab training notebook | No — standalone |
| `SETUP.md` | Complete environment setup guide | New doc |

### Files modified

| File | What changed |
|------|-------------|
| `config/settings.json` | Added `recording` and `inference` sections |
| `.gitignore` | Added sections for recordings, model weights, notebooks |
| `requirements.txt` | Updated versions, added torch, swapped to onnxruntime-gpu |
| `start_wsl.sh` | Fixed E: → D: drive path |
| `src/capture.py` | Added 50ms sleep to MediaPipe thread (reduces CPU strain) |

### Files NOT modified (original pipeline untouched)

| File | Status |
|------|--------|
| `run_windows.py` | ✅ Unchanged — live demo still works exactly as before |
| `src/capture.py` | ✅ Core logic unchanged — only added a `time.sleep(0.05)` |
| `src/mapper.py` | ✅ Unchanged — WSL normalization code untouched |
| `src/bridge.py` | ✅ Unchanged — Godot UDP bridge untouched |
| `src/send_to_wsl.py` | ✅ Unchanged — Windows→WSL bridge untouched |
| `src/main_wsl.py` | ✅ Unchanged — WSL processing layer untouched |
| `src/utils.py` | ✅ Unchanged |
| `src/utils_wsl.py` | ✅ Unchanged |
| `khmer-sign-mannequin2/` | ✅ Unchanged — Godot project untouched |

---

## 3. File Map — What Every File Does

```
khmer_sign_recognizer/
│
├── ── TOP-LEVEL SCRIPTS ──────────────────────────────────────────
│
├── run_windows.py          PIPELINE A entry point
│                           Camera → skeleton → send UDP to WSL
│                           Used for: live demo with Godot mannequin
│
├── record_signs.py         PIPELINE B entry point
│                           Camera → skeleton → save .npy recordings
│                           Used for: collecting training data
│
├── inference.py            PIPELINE B inference
│                           Camera → skeleton → model predicts sign
│                           Used for: live sign-to-text (after training)
│
├── start_windows.bat       Shortcut: activates venv + runs run_windows.py
├── start_wsl.sh            Shortcut: activates venv_wsl + runs main_wsl.py
│
├── ── SOURCE CODE ────────────────────────────────────────────────
│
├── src/
│   ├── capture.py          SHARED — dual pipeline camera capture
│   │                       RTMPose (body/GPU) + MediaPipe (hands/CPU)
│   │                       Used by: run_windows.py, record_signs.py, inference.py
│   │
│   ├── ── Pipeline A (Live Demo) ──
│   ├── send_to_wsl.py      Sends landmarks from Windows → WSL via UDP
│   ├── main_wsl.py         WSL receiver — normalizes + forwards to Godot
│   ├── mapper.py           WSL normalizer — shoulder-width scaling
│   ├── bridge.py           Sends bone transforms from WSL → Godot via UDP
│   ├── utils.py            Windows-side config + logging
│   ├── utils_wsl.py        WSL-side config + logging + math helpers
│   │
│   ├── ── Pipeline B (AI Training) ──
│   ├── normalizer.py       Local normalization (no WSL needed)
│   │                       Produces clean (normalized) + raw (noisy) versions
│   ├── recorder.py         Buffers frames → saves .npy files
│   │                       Handles pad/trim to 60 frames
│   ├── model.py            Temporal CNN architecture (PyTorch)
│   │                       Same code runs locally + in Colab
│   ├── dataset.py          PyTorch Dataset — auto-discovers label folders
│   └── visualizer.py       2D stick figure replay for .npy files
│
├── ── CONFIG ─────────────────────────────────────────────────────
│
├── config/
│   └── settings.json       All settings: camera, mediapipe, network,
│                           recording, inference, smoothing, logging
│
├── ── DATA (git-ignored, local only) ─────────────────────────────
│
├── data/
│   └── sequences/          Recorded .npy files, organized by label
│       ├── hello/
│       │   ├── hello_001_clean.npy    (60, 51, 3) normalized
│       │   └── hello_001_raw.npy      (60, 51, 3) raw pixel coords
│       └── water/
│           └── ...
│
├── models/
│   └── weights/            Trained model .pth files from Colab
│       └── ksl_model_latest.pth
│
├── ── NOTEBOOKS (for Colab) ───────────────────────────────────────
│
├── notebooks/
│   └── train_ksl.py        Self-contained training script
│                           Uses # %% cell markers for Colab
│
├── ── GODOT (3D mannequin for demos) ─────────────────────────────
│
├── khmer-sign-mannequin2/
│   ├── main.gd             Receives UDP bone data, animates Y-Bot
│   ├── main.tscn           Scene with Y-Bot model
│   └── ...
│
├── ── DOCS ───────────────────────────────────────────────────────
│
├── README.md               Original project readme
├── SETUP.md                Environment setup guide (venv + WSL)
│
├── ── ENVIRONMENT (git-ignored) ──────────────────────────────────
│
├── venv/                   Windows Python 3.12 virtual env
├── venv_wsl/               WSL Python virtual env
├── requirements.txt        Python dependencies
└── .gitignore              Excludes recordings, models, venvs, logs
```

---

## 4. The Two Pipelines Explained

### Pipeline A — Live Demo (Original System)

**When to use:** Presentations, demos, showing the mannequin moving.

**What runs:**
```
Terminal 1 (WSL):     python src/main_wsl.py    ← listens for UDP
Terminal 2 (Windows): python run_windows.py     ← captures + sends
Godot:                Open khmer-sign-mannequin2 project + run
```

**Data flow:**
```
Webcam → RTMPose (body) + MediaPipe (hands)
  → JSON over UDP to WSL (port 9999)
  → WSL normalizes (mapper.py)
  → JSON over UDP to Godot (port 8888)
  → Godot animates Y-Bot mannequin
```

### Pipeline B — AI Training (New System)

**When to use:** Recording data, training the model, running inference.

**Step 1: Record data**
```powershell
python record_signs.py
```
- Type a sign label (e.g. `hello`), press ENTER
- Press SPACE → perform sign → press SPACE (or wait 60 frames)
- Saves `hello_001_clean.npy` + `hello_001_raw.npy`
- No WSL, no Godot, no network — everything runs locally

**Step 2: Verify recording with stick figure**
```powershell
python -m src.visualizer data/sequences/hello/hello_001_clean.npy
```
- Opens a window showing a 2D stick figure replaying your recording
- Green lines = body, blue = left hand, red = right hand
- Press R to replay, Q to quit

**Step 3: Train (Google Colab — later)**
```
Upload data/sequences/ to Google Drive → Open train_ksl.py in Colab → Run All
→ Downloads ksl_model_v1.pth from Drive → Place in models/weights/
```

**Step 4: Run inference**
```powershell
python inference.py
```
- Uses the same webcam + capture pipeline
- Feeds live frames through the trained model
- Shows predicted sign name on screen

---

## 5. How to Run Everything

### First-time setup (already done)

```powershell
cd "D:\Projects\Sign to Text\khmer_sign_recognizer"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Demo with Godot mannequin (Pipeline A)

```
# Terminal 1 — WSL receiver (start first)
wsl
cd "/mnt/d/Projects/Sign to Text/khmer_sign_recognizer"
source venv_wsl/bin/activate
python src/main_wsl.py

# Terminal 2 — Windows capture
.\venv\Scripts\Activate.ps1
python run_windows.py

# Terminal 3 — Godot
Open khmer-sign-mannequin2 in Godot editor → press Play
```

### Record training data (Pipeline B)

```powershell
.\venv\Scripts\Activate.ps1
python record_signs.py
```

### Replay a recording as stick figure

```powershell
python -m src.visualizer data/sequences/hello/hello_001_clean.npy
```

### Run live inference (after training)

```powershell
.\venv\Scripts\Activate.ps1
python inference.py
```

---

## 6. How to Add New Features Without Breaking Old Ones

### The golden rule: new files, not modified files

Every new feature we added follows this pattern:

```
DON'T: edit capture.py to add recording logic
DO:    create recorder.py that IMPORTS from capture.py

DON'T: edit mapper.py to add local normalization
DO:    create normalizer.py that COPIES the math from mapper.py

DON'T: edit run_windows.py to add sign prediction
DO:    create inference.py that uses the same capture.py
```

**Why?** If `run_windows.py` still works exactly like before, you know Pipeline A is safe. If you break something in the new code, Pipeline A keeps working.

### How to add a new feature (pattern to follow)

1. **Create a new file in `src/`** for the logic (e.g. `src/my_feature.py`)
2. **Import from existing files**, don't modify them
3. **Create a new top-level script** if it's a new mode (e.g. `my_feature_runner.py`)
4. **Add config** in `config/settings.json` under a new section name
5. **Don't touch** `run_windows.py`, `capture.py`, `mapper.py`, or `bridge.py`

### When it IS okay to modify existing files

- **`config/settings.json`** — add new sections (don't rename existing keys)
- **`.gitignore`** — add new ignore patterns
- **`requirements.txt`** — add new packages (don't remove existing ones unless replacing)
- **`capture.py`** — ONLY for performance fixes (like the `time.sleep` we added)

### Config structure

Settings are namespaced so they don't collide:

```json
{
  "capture":    { ... },     ← Pipeline A + B (shared)
  "mediapipe":  { ... },     ← Pipeline A + B (shared)
  "network":    { ... },     ← Pipeline A only
  "smoothing":  { ... },     ← Pipeline A (WSL mapper)
  "recording":  { ... },     ← Pipeline B (recording)
  "inference":  { ... },     ← Pipeline B (inference)
  "logging":    { ... }      ← Both
}
```

To add a new feature's config, just add a new top-level key:
```json
"my_new_feature": {
    "setting1": "value1"
}
```

---

## 7. What Goes in Git vs What Stays Local

### ✅ GOES IN GIT (shared with team via repo)

| What | Why |
|------|-----|
| All `.py` source files | Code is shared |
| `config/settings.json` | Default config is shared |
| `requirements.txt` | Dependencies are shared |
| `.gitignore` | Ignore rules are shared |
| `README.md`, `SETUP.md` | Docs are shared |
| `notebooks/train_ksl.py` | Training notebook is shared |
| `khmer-sign-mannequin2/` | Godot project is shared |
| `.bat` and `.sh` scripts | Startup scripts are shared |

### 🚫 NEVER GOES IN GIT (stays on your machine)

| What | Why | Gitignore rule |
|------|-----|----------------|
| `venv/`, `venv_wsl/` | Virtual envs have hardcoded paths | `venv/`, `venv_wsl/` |
| `data/sequences/**/*.npy` | Your personal recordings | `data/`, `*.npy` |
| `models/weights/*.pth` | Trained model weights (large binary files) | `models/`, `*.pth` |
| `logs/*.log` | Runtime logs | `logs/`, `*.log` |
| `.env` | API keys, secrets | `.env` |
| `__pycache__/` | Python bytecode cache | `__pycache__/`, `*.pyc` |

**Your recordings are always private.** The `data/` folder is git-ignored entirely. Even if you accidentally `git add .`, none of your `.npy` files will be included.

---

## 8. Dependency Management

### Windows venv (`venv\`)

```
opencv-python==4.13.0.92      Camera + display
mediapipe==0.10.14             Hand tracking (DO NOT UPGRADE past 0.10.14)
rtmlib==0.0.15                 RTMPose body tracking
onnxruntime-gpu                ONNX inference on GPU
numpy==2.4.4                   Math (fallback: 1.26.4)
python-json-logger==4.1.0     Structured logging
torch                          PyTorch for model inference
```

Install: `pip install -r requirements.txt`
PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### WSL venv (`venv_wsl/`)

```
numpy==2.4.4
python-json-logger==4.1.0
```

The WSL side only does math + JSON — no heavy packages needed.

### Adding a new dependency

1. `pip install packagename` inside the activated venv
2. Add it to `requirements.txt` with a comment explaining what it's for
3. Pin the version if it matters: `packagename==1.2.3`

---

## 9. Known Issues & Future Work

### Known Issues

| Issue | Status | Notes |
|-------|--------|-------|
| MediaPipe locked to 0.10.14 | ⚠️ Permanent | Google removed Holistic API in 0.10.18 |
| RTMPose falls back to CPU if `onnxruntime-gpu` not installed | Fixed | Swapped to `onnxruntime-gpu` in requirements |
| venvs break if project moves to another drive | Expected | Delete and recreate per SETUP.md |

### Future Work (not built yet)

| Feature | Status | When |
|---------|--------|------|
| Google Drive + Colab integration | Notebook ready, setup pending | After recording data |
| Federated learning (teammate weight merging) | Designed, not coded | Year 3 |
| Variable-length sequence support (LSTM/Transformer) | Not started | After Temporal CNN proves out |
| Real-time text-to-speech for recognized signs | Not started | After inference works |
