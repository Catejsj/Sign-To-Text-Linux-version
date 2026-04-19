# 🛠️ Khmer Sign Recognizer — Full Environment Setup Guide

> **Why this doc exists:** The project was moved from an external SSD (`E:\`) to `D:\Projects\Sign to Text\`.
> Virtual environments have hardcoded paths and **cannot be moved** — they must be rebuilt fresh.

---

## 🏗️ How the System Works

This project runs as **two separate layers** that talk to each other over UDP:

```
[Windows Layer]                     [WSL Layer]                    [Godot]
  camera + mediapipe/rtmlib   →UDP→  skeleton mapper + AI brain  →UDP→  3D model
  run_windows.py                     src/main_wsl.py
  venv\  (Windows Python)            venv_wsl/  (Linux Python)
```

You need **two virtual environments**:
- `venv\` — Windows side (PowerShell)
- `venv_wsl/` — WSL/Linux side (Bash terminal)

---

## ✅ Prerequisites

Make sure you have these installed **before** starting:

| Tool | Where | Check |
|------|-------|-------|
| Python 3.12 | Windows | `python --version` in PowerShell |
| Python 3 | WSL | `python3 --version` in WSL terminal |
| WSL 2 | Windows | `wsl --version` in PowerShell |
| pip | Both | `pip --version` |

> **Python install path (yours):** `C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe`
> If Python is missing: https://www.python.org/downloads/

---

# 🪟 PART 1 — Windows Side (`venv\`)

## Step 1 — Open PowerShell in the Project Folder

```powershell
cd "D:\Projects\Sign to Text\khmer_sign_recognizer"
```

## Step 2 — Create the Windows Virtual Environment

```powershell
python -m venv venv
```

## Step 3 — Activate the Windows Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

Your prompt will show `(venv)` when active:
```
(venv) PS D:\Projects\Sign to Text\khmer_sign_recognizer>
```

> **If you get a script execution error, run this once then retry:**
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

## Step 4 — Install Windows Dependencies

```powershell
pip install -r requirements.txt
```

## Step 5 — Verify Windows Install

```powershell
python -c "import cv2; import mediapipe; import rtmlib; import onnxruntime; import numpy; print('Windows side: All good!')"
```

---

# 🐧 PART 2 — WSL Side (`venv_wsl/`)

## Step 1 — Open a WSL Terminal

In PowerShell or Windows Terminal, type:
```powershell
wsl
```

## Step 2 — Navigate to the Project (via D: drive mount)

```bash
cd "/mnt/d/Projects/Sign to Text/khmer_sign_recognizer"
```

> ✅ The `start_wsl.sh` has already been updated to use this correct path.

## Step 3 — Install Python venv support (if missing)

```bash
sudo apt update && sudo apt install python3-venv python3-pip -y
```

## Step 4 — Create the WSL Virtual Environment

```bash
python3 -m venv venv_wsl
```

## Step 5 — Activate the WSL Virtual Environment

```bash
source venv_wsl/bin/activate
```

Your prompt will show `(venv_wsl)` when active:
```
(venv_wsl) user@machine:/mnt/d/Projects/Sign to Text/khmer_sign_recognizer$
```

## Step 6 — Install WSL Dependencies

> ⚠️ **IMPORTANT — Make sure `(venv_wsl)` is showing in your prompt before running pip.**
> If you see the error `externally-managed-environment`, it means you skipped activation.
> Run `source venv_wsl/bin/activate` first, then retry.

The WSL side only needs the processing/math packages (no camera/mediapipe needed):

```bash
pip install numpy==2.4.4 python-json-logger==4.1.0
```

> The WSL layer (`main_wsl.py`) only uses `numpy`, `json`, `socket`, and `logging` — no opencv/mediapipe needed there.

## Step 7 — Verify WSL Install

```bash
python -c "import numpy; print('WSL side: All good!')"
```

---

# 🚀 Running the Full System

### Terminal 1 — Start WSL layer first (it listens for UDP)

```bash
# In WSL terminal
cd "/mnt/d/Projects/Sign to Text/khmer_sign_recognizer"
source venv_wsl/bin/activate
python src/main_wsl.py
```

Or just run the shell script:
```bash
bash start_wsl.sh
```

### Terminal 2 — Start Windows layer

```powershell
# In PowerShell
cd "D:\Projects\Sign to Text\khmer_sign_recognizer"
.\venv\Scripts\Activate.ps1
python run_windows.py
```

Or double-click `start_windows.bat`.

---

# 🔄 Daily Cheat Sheet

**Windows (PowerShell):**
```powershell
cd "D:\Projects\Sign to Text\khmer_sign_recognizer"
.\venv\Scripts\Activate.ps1     # activate
deactivate                       # deactivate when done
```

**WSL (bash):**
```bash
cd "/mnt/d/Projects/Sign to Text/khmer_sign_recognizer"
source venv_wsl/bin/activate    # activate
deactivate                       # deactivate when done
```

---

# 🗑️ Starting Over (nuke both envs)

**Windows (PowerShell):**
```powershell
Remove-Item -Recurse -Force "D:\Projects\Sign to Text\khmer_sign_recognizer\venv"
```

**WSL (bash):**
```bash
rm -rf "/mnt/d/Projects/Sign to Text/khmer_sign_recognizer/venv_wsl"
```

---

# 📋 Latest Package Versions (April 2026)

| Package | Old (pinned) | Latest | Side | Notes |
|---------|-------------|--------|------|-------|
| `opencv-python` | 4.10.0.84 | **4.13.0.92** | Windows | Safe to upgrade |
| `mediapipe` | 0.10.14 | **0.10.14** ⚠️ STAY HERE | Windows | 0.10.18+ removed Holistic — DO NOT upgrade |
| `rtmlib` | unpinned | **0.0.15** | Windows | Safe to use latest |
| `onnxruntime` | unpinned | **1.24.4** | Windows | Safe to use latest |
| `numpy` | 1.26.4 | **2.4.4** | Both | ⚠️ Fallback to 1.26.4 if errors |
| `python-json-logger` | 2.0.7 | **4.1.0** | Both | Safe to upgrade |
