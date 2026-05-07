# 🛠️ SignLink — Full Setup Guide (From Scratch)

> **Who is this for?** A teammate setting up the project for the first time on a Windows PC.
> This guide covers everything: installing prerequisites, cloning the code, building both virtual environments, configuring Godot, and running the live mannequin demo.

> **Last updated:** April 2026

---

## 🏗️ How the System Works

The project runs as **three separate programs** on the same machine, talking over UDP:

```
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│  WINDOWS LAYER   │  →UDP→  │   WSL LAYER      │  →UDP→  │   GODOT LAYER    │
│  run_windows.py  │  :9999  │  src/main_wsl.py │  :8888  │   main.gd        │
│                  │         │                  │         │                  │
│  Camera + AI     │         │  Skeleton math   │         │  3D mannequin    │
│  (RTMPose + MP)  │         │  + smoothing     │         │  (Y-Bot)         │
│                  │         │                  │         │                  │
│  venv\           │         │  venv_wsl/       │         │  Godot 4.6       │
│  (Win Python)    │         │  (Linux Python)  │         │  (game engine)   │
└──────────────────┘         └──────────────────┘         └──────────────────┘
```

You'll set up **two Python virtual environments** (one for Windows, one for WSL) and install **Godot** separately.

---

# 📋 PART 0 — Prerequisites

Install all of these **before** continuing.

## 0.1 — Python 3.12 (Windows)

> ⚠️ **You MUST use Python 3.12 or lower (3.9–3.12).** Python 3.13+ will NOT work because `mediapipe==0.10.14` does not have wheels for it. If you already have Python 3.13 installed, you can install 3.12 alongside it — they coexist fine.

1. Download **Python 3.12** specifically from https://www.python.org/downloads/release/python-3120/
2. **CHECK** ✅ "Add Python to PATH" during install
3. Verify:
   ```powershell
   python --version
   # Should print: Python 3.12.x
   ```
   If it shows 3.13+, use the `py` launcher to check what's available:
   ```powershell
   py --list
   # Should show -3.12-64 somewhere in the list
   ```

## 0.2 — Git

1. Download from https://git-scm.com/downloads/win
2. Install with defaults (keep "Git from the command line" checked)
3. Verify:
   ```powershell
   git --version
   ```

## 0.3 — WSL 2 (Windows Subsystem for Linux)

1. Open **PowerShell as Administrator** and run:
   ```powershell
   wsl --install
   ```
   This installs WSL 2 + Ubuntu by default. **Restart your PC** when prompted.

2. After reboot, Ubuntu will open automatically and ask you to create a **username + password**. Choose something simple — you'll need the password for `sudo`.

3. Verify:
   ```powershell
   wsl --version
   # Should show WSL version: 2.x.x
   ```

> **Already have WSL?** Make sure it's version 2:
> ```powershell
> wsl --set-default-version 2
> ```

## 0.4 — Godot 4.6

1. Download **Godot 4.6** (standard version, not .NET) from https://godotengine.org/download
2. Extract it anywhere you like (e.g., `C:\Godot\`)
3. No installer — just run the `.exe` when needed

## 0.5 — NVIDIA GPU Driver (Recommended)

If you have an NVIDIA GPU, make sure your driver is up to date for CUDA/ONNX acceleration:
- Download from https://www.nvidia.com/Download/index.aspx
- The Windows capture layer uses `onnxruntime-gpu` to run RTMPose on your GPU

> **No NVIDIA GPU?** The system will still work using CPU-only inference. It'll just be slower.

---

# 📥 PART 1 — Clone the Repository

## Step 1 — Pick a location

Choose where you want the project. This guide uses `D:\Projects\` — adjust if needed.

```powershell
# Create the folder if it doesn't exist
mkdir "D:\Projects" -Force
```

## Step 2 — Clone

```powershell
cd "D:\Projects"
git clone https://github.com/Catejsj/Sign-to-Text.git "Sign to Text"
```

## Step 3 — Verify the structure

```powershell
cd "D:\Projects\Sign to Text\khmer_sign_recognizer"
dir
```

You should see files like `run_windows.py`, `start_wsl.sh`, `requirements.txt`, the `src/` folder, etc.

---

# 🪟 PART 2 — Windows Environment Setup (`venv\`)

## Step 1 — Open PowerShell in the project folder

```powershell
cd "D:\Projects\Sign to Text\khmer_sign_recognizer"
```

## Step 2 — Create the virtual environment

```powershell
# If Python 3.12 is your default:
python -m venv venv

# If you have multiple Python versions (e.g. 3.13 is default but 3.12 is also installed):
py -3.12 -m venv venv
```

> **How to tell?** Run `python --version`. If it says 3.13+, use the `py -3.12` command instead.

## Step 3 — Activate it

```powershell
.\venv\Scripts\Activate.ps1
```

Your prompt should now show `(venv)`:
```
(venv) PS D:\Projects\Sign to Text\khmer_sign_recognizer>
```

> **⚠️ Script execution error?** Run this once, then retry:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

## Step 4 — Install dependencies

```powershell
pip install -r requirements.txt
```

This installs:
| Package | Purpose |
|---------|---------|
| `opencv-python` | Webcam capture + display |
| `mediapipe==0.10.14` | Hand tracking (21 joints per hand) |
| `rtmlib` | Body tracking via RTMPose |
| `onnxruntime-gpu` | GPU-accelerated inference for RTMPose |
| `numpy` | Math / array processing |
| `python-json-logger` | Structured logging |
| `torch` | PyTorch for AI model inference |

> **⚠️ CRITICAL:** Do NOT upgrade `mediapipe` past `0.10.14`. Version `0.10.18+` removed the Holistic API that we depend on.
>
> **Getting "Could not find a version that satisfies the requirement mediapipe==0.10.14"?**
> This means your Python is 3.13+. You need Python 3.12. Go back to Part 0, install Python 3.12, then nuke and rebuild the venv:
> ```powershell
> Remove-Item -Recurse -Force venv
> py -3.12 -m venv venv
> .\venv\Scripts\Activate.ps1
> pip install -r requirements.txt
> ```

### Optional: Install PyTorch with CUDA (for local training / GPU inference)

If you have an NVIDIA GPU and want GPU-accelerated PyTorch:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Step 5 — Verify

```powershell
python -c "import cv2; import mediapipe; import rtmlib; import onnxruntime; import numpy; print('Windows side: All good!')"
```

If you see `Windows side: All good!` — you're done with Windows. ✅

---

# 🐧 PART 3 — WSL Environment Setup (`venv_wsl/`)

## Step 1 — Open a WSL terminal

From PowerShell or Windows Terminal:
```powershell
wsl
```

You're now in Linux (Ubuntu).

## Step 2 — Install Python venv support

```bash
sudo apt update && sudo apt install python3-venv python3-pip -y
```

## Step 3 — Navigate to the project

WSL accesses your Windows drives through `/mnt/`:

```bash
cd "/mnt/d/Projects/Sign to Text/khmer_sign_recognizer"
```

> **Different drive letter?** Replace `/mnt/d/` with `/mnt/c/`, `/mnt/e/`, etc.

## Step 4 — Create the WSL virtual environment

```bash
python3 -m venv venv_wsl
```

## Step 5 — Activate it

```bash
source venv_wsl/bin/activate
```

Your prompt should now show `(venv_wsl)`:
```
(venv_wsl) user@DESKTOP-XXX:/mnt/d/Projects/Sign to Text/khmer_sign_recognizer$
```

## Step 6 — Install WSL dependencies

> **⚠️ Make sure `(venv_wsl)` is showing in your prompt.** If you see the error `externally-managed-environment`, you forgot to activate. Run `source venv_wsl/bin/activate` first.

The WSL side only needs lightweight packages (no camera / no AI models):

```bash
pip install numpy==2.4.4 python-json-logger==4.1.0
```

> **Why so few packages?** The WSL layer (`main_wsl.py`) only does math and networking — it uses `numpy`, `json`, `socket`, and `logging`. No OpenCV or MediaPipe needed.

## Step 7 — Verify

```bash
python -c "import numpy; print('WSL side: All good!')"
```

If you see `WSL side: All good!` — you're done with WSL. ✅

---

# 🎮 PART 4 — Godot Setup

## Step 1 — Open Godot

Launch the Godot 4.6 executable you downloaded earlier.

## Step 2 — Import the project

1. In the Godot Project Manager, click **Import**
2. Navigate to:
   ```
   D:\Projects\Sign to Text\khmer_sign_recognizer\khmer-sign-mannequin2\
   ```
3. Select the `project.godot` file
4. Click **Import & Edit**

## Step 3 — Verify

You should see the 3D scene with the Y-Bot mannequin. Don't press F5 yet — we need to fix the network IPs first.

---

# 🔌 PART 5 — Network Configuration (IP Addresses)

WSL 2 gets a **new virtual IP address** every time you reboot your PC. The three layers need to know each other's IPs to communicate over UDP. We have a script that fixes this automatically.

## Step 1 — Run the IP updater

In **PowerShell** (not WSL), with the Windows venv active:

```powershell
cd "D:\Projects\Sign to Text\khmer_sign_recognizer"
.\venv\Scripts\Activate.ps1
python update_ips.py
```

You should see output like:
```
INFO: ✅ Found WSL IP: 172.19.XX.XX
INFO: ✅ Found Windows Host IP (from WSL): 172.19.XX.XX
INFO: ✅ Successfully updated config/settings.json
```

> **⚠️ You need to run `update_ips.py` every time you reboot your PC!** The WSL IP changes on each boot.

## Step 2 — Verify the config

Open `config/settings.json` and check the `network` section:
```json
{
  "network": {
    "wsl_ip": "172.19.XX.XX",     ← should NOT be 127.0.0.1
    "wsl_port": 9999,
    "receive_port": 9999,
    "godot_ip": "172.19.XX.XX",   ← should NOT be 127.0.0.1
    "godot_port": 8888,
    "protocol": "udp"
  }
}
```

If the IPs look like real addresses (not `127.0.0.1`), you're good.

---

# 🚀 PART 6 — Running the Full System

> **Start in this exact order.** Listeners first, sender last.

### Terminal 1 — Start Godot (listens on port 8888)

1. Open Godot → load the `khmer-sign-mannequin2` project
2. Press **F5** to run the scene
3. You should see the mannequin standing idle and the console saying:
   ```
   ✅ Skeleton found
   ✅ UDP listening on port 8888
   Waiting for bone data from WSL...
   ```

### Terminal 2 — Start WSL layer (listens on port 9999)

Open a WSL terminal:
```bash
cd "/mnt/d/Projects/Sign to Text/khmer_sign_recognizer"
source venv_wsl/bin/activate
python src/main_wsl.py
```

Or use the shortcut script:
```bash
bash start_wsl.sh
```

### Terminal 3 — Start Windows layer (sends data)

Open a **new PowerShell** window:
```powershell
cd "D:\Projects\Sign to Text\khmer_sign_recognizer"
.\venv\Scripts\Activate.ps1
python run_windows.py
```

Or just double-click `start_windows.bat`.

### ✅ You should now see:

- A **camera window** showing your webcam feed with skeleton overlay
- The camera shows **"CAPTURING"** in green text when your body is detected
- The **Godot mannequin** moving its arms and fingers in real-time, mirroring you!

> **Controls in the camera window:**
> - `Q` — Quit
> - `F` — Fullscreen
> - `N` — Normal window

---

# 🔄 Daily Cheat Sheet

**Every time you reboot and want to run the system:**

```powershell
# Step 0 — Fix IPs (PowerShell, do this FIRST)
cd "D:\Projects\Sign to Text\khmer_sign_recognizer"
.\venv\Scripts\Activate.ps1
python update_ips.py

# Step 1 — Start Godot
# Open Godot → khmer-sign-mannequin2 → F5

# Step 2 — Start WSL (new terminal)
wsl
cd "/mnt/d/Projects/Sign to Text/khmer_sign_recognizer"
bash start_wsl.sh

# Step 3 — Start Windows (new PowerShell)
cd "D:\Projects\Sign to Text\khmer_sign_recognizer"
.\venv\Scripts\Activate.ps1
python run_windows.py
```

**Activate / deactivate venvs (for other work):**

```powershell
# Windows
.\venv\Scripts\Activate.ps1     # activate
deactivate                       # deactivate
```

```bash
# WSL
source venv_wsl/bin/activate    # activate
deactivate                       # deactivate
```

---

# 🧯 Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| Mannequin doesn't move at all | IPs are stale (PC rebooted) | Run `python update_ips.py` |
| Camera opens but mannequin frozen | WSL or Godot isn't running | Start them in correct order (Godot → WSL → Windows) |
| `Failed to open camera` | Another app using webcam | Close Zoom / Teams / Discord / OBS |
| Camera says "NOT CAPTURING" | No body detected for 30+ frames | Stand in front of camera with shoulders visible |
| Fingers don't move | Hands not visible to camera | Hold hands clearly in frame |
| Arms are jittery | Smoothing needs tuning | Adjust `min_cutoff` and `beta` in `config/settings.json` |
| `externally-managed-environment` error in WSL | Forgot to activate venv | Run `source venv_wsl/bin/activate` first |
| PowerShell script execution error | Execution policy blocks .ps1 | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `No module named 'rtmlib'` | Wrong venv or install failed | Activate the Windows venv and re-run `pip install -r requirements.txt` |
| WSL can't find the project folder | Wrong mount path | Check your drive letter: `ls /mnt/` to see available drives |
| `wsl --install` hangs or fails | Need Windows 10 v2004+ or Windows 11 | Update Windows via Settings → Windows Update |
| `Could not find a version that satisfies mediapipe==0.10.14` | Python 3.13+ installed (no wheel exists) | Install **Python 3.12**, recreate venv with `py -3.12 -m venv venv` |

---

# 🗑️ Starting Over (Nuke & Rebuild)

If something goes wrong and you want to rebuild the virtual environments from scratch:

**Windows (PowerShell):**
```powershell
Remove-Item -Recurse -Force "D:\Projects\Sign to Text\khmer_sign_recognizer\venv"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**WSL (bash):**
```bash
rm -rf "/mnt/d/Projects/Sign to Text/khmer_sign_recognizer/venv_wsl"
python3 -m venv venv_wsl
source venv_wsl/bin/activate
pip install numpy==2.4.4 python-json-logger==4.1.0
```

---

# 📋 Package Reference (April 2026)

| Package | Version | Side | Notes |
|---------|---------|------|-------|
| `opencv-python` | `4.13.0.92` | Windows | Webcam + display |
| `mediapipe` | `0.10.14` | Windows | ⚠️ **DO NOT UPGRADE** — 0.10.18+ breaks Holistic |
| `rtmlib` | `0.0.15` | Windows | Body pose via RTMPose |
| `onnxruntime-gpu` | latest | Windows | GPU inference for RTMPose |
| `numpy` | `2.4.4` | Both | ⚠️ Fallback to `1.26.4` if anything breaks |
| `python-json-logger` | `4.1.0` | Both | Structured logging |
| `torch` | latest | Windows | AI model inference |

---

# 📁 Key Files Reference

```
khmer_sign_recognizer/
│
├── config/settings.json          ← Network IPs, camera settings, filter tuning
├── update_ips.py                 ← Run after every reboot to fix WSL IPs
├── requirements.txt              ← Windows Python dependencies
│
├── start_windows.bat             ← Double-click to launch Windows layer
├── run_windows.py                ← Windows entry point (camera + AI)
│
├── start_wsl.sh                  ← Run in WSL terminal to launch WSL layer
│
├── src/
│   ├── capture.py                ← Camera + RTMPose + MediaPipe
│   ├── send_to_wsl.py            ← UDP sender: Windows → WSL
│   ├── main_wsl.py               ← WSL entry point (skeleton math)
│   ├── mapper.py                 ← Normalizes + bone rotation calc
│   ├── bridge.py                 ← UDP sender: WSL → Godot
│   ├── utils.py                  ← Windows utilities
│   └── utils_wsl.py              ← WSL utilities (quaternions, filters)
│
└── khmer-sign-mannequin2/
    ├── project.godot             ← Open this in Godot 4.6
    └── main.gd                   ← Receives UDP data, moves the Y-Bot
```

---

> **Questions?** Ask in the group chat or check `WORKFLOW.md` for the full data collection + training cycle.
