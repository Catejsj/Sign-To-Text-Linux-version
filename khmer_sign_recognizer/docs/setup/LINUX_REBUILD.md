# Rebuilding SignLink on Linux (CachyOS / Arch)

The exact procedure to bring the project back to a working state after switching
from Windows to Linux. The code is cross-platform Python; the environment is the
part that needs care. Written for CachyOS (Arch-based), but any Arch distro works.

Verified working on CachyOS with an RTX 4060 Laptop GPU (driver 610.43.03).
Package versions that worked are frozen in `requirements-lock-linux.txt`
(Windows equivalents are in `requirements-lock-windows.txt`).

---

## 0. System prerequisites (once)

```bash
sudo pacman -S --needed git base-devel noto-fonts

# Python 3.11 — REQUIRED, see below. CachyOS ships Python 3.14 as `python`.
sudo pacman -S --needed python311

# verify the GPU is visible (driver only — see "No system CUDA" below)
nvidia-smi          # should list the RTX 4060
```

### Python 3.11 is not optional
CachyOS's system `python` is **3.14**. `mediapipe` 0.10.14 publishes no wheels
past Python 3.12, and `open3d` is similar. A venv built on 3.14 cannot install
the stack at all. Everything below assumes `python3.11`.

### No system CUDA toolkit is needed
Do **not** `pacman -S cuda cudnn`. The CUDA libraries come from the PyTorch
wheel (`site-packages/nvidia/*/lib`), and onnxruntime is pointed at those (see
step 3). Only the NVIDIA **driver** is required, which CachyOS installs already.

---

## 1. Get the project

```bash
git clone https://github.com/Catejsj/Sign-to-Text.git
cd Sign-to-Text/khmer_sign_recognizer
```

If copying the SSD folder instead of cloning and git complains about
"dubious ownership":

```bash
git config --global --add safe.directory '<full path to the repo>'
```

---

## 2. Create the virtual environment

```bash
python3.11 -m venv venv          # NOT `python -m venv` — that is 3.14
source venv/bin/activate         # Linux; not venv\Scripts\Activate.ps1
```

---

## 3. Install dependencies (order matters)

**Install torch FIRST, from the cu12 index.** The default PyPI torch now ships
**CUDA 13** libraries, but `onnxruntime-gpu` is built against **CUDA 12**
(`libcublas.so.12`, `libcudnn.so.9`). Mismatched sonames mean the CUDA provider
can never load, and RTMPose silently runs on CPU.

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then the rest:

```bash
pip install -r requirements-linux.txt
```

### Then remove the CPU onnxruntime — this step is easy to miss
`rtmlib` depends on `onnxruntime` (the **CPU** package). pip installs it *over*
`onnxruntime-gpu` — same `onnxruntime/` directory — which strips CUDA out
entirely. After any `pip install -r ...`, check and repair:

```bash
pip list | grep onnxruntime          # if plain `onnxruntime` is listed:
pip uninstall -y onnxruntime
pip install --force-reinstall "onnxruntime-gpu==1.24.4"
```

(The `rtmlib requires onnxruntime, which is not installed` warning afterwards is
harmless — `onnxruntime-gpu` provides the same module; pip just can't tell.)

### Do NOT downgrade numpy
Older notes said to pin `numpy==1.26.4` if mediapipe complains. Don't:
`opencv-python` 4.13 hard-requires `numpy>=2`, so that pin makes the install
unresolvable. mediapipe 0.10.14 runs fine against numpy 2.4.4 (as it did on
Windows).

---

## 4. Verify everything works

The GPU checks matter most. **`onnxruntime.get_available_providers()` is not
proof** — it lists what the build supports, not what can actually load. A session
will happily bind CPU and return correct (slow) results with only a warning.

```bash
# 1. torch sees the GPU  -> must print True
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 2. onnxruntime can REALLY put a session on the GPU. Raises if it cannot.
python -c "import sys; sys.path.insert(0,'.'); from src.cuda_setup import assert_onnx_gpu; assert_onnx_gpu(); print('ONNX GPU OK')"

# 3. core imports load
python -c "import cv2, mediapipe, sklearn, open3d, rtmlib; print('imports OK')"
```

`src/cuda_setup.py` is what makes (2) pass: it dlopens the CUDA libraries out of
`site-packages/nvidia/*/lib` with `RTLD_GLOBAL` before the first ONNX session, so
onnxruntime can resolve them. `src/capture.py` and `scripts/compare_detectors.py`
call it automatically. Note `LD_LIBRARY_PATH` cannot fix this from inside Python
— glibc reads it once at process start.

### Camera
OpenCV opens `/dev/video0` via **V4L2**; the code picks the backend per platform
(`CAP_DSHOW` is Windows-only and was hard-coded before).

### 3D mannequin (Open3D) on Wayland
Open3D renders through GLFW, and GLFW's Wayland path fails to initialize GLEW —
`create_window()` returns False and no window appears. `scripts/mannequin_local.py`
now steers GLFW to X11/XWayland automatically. (Older notes suggested
`QT_QPA_PLATFORM=xcb`; that does nothing here — Open3D uses GLFW, not Qt.)

If you ever need it manually:

```bash
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 DISPLAY=:0 python scripts/mannequin_local.py ...
```

### Khmer font
`fonts/NotoSansKhmer-Regular.ttf` ships in the repo and is found first, so Khmer
renders with no extra setup. `noto-fonts` provides a system fallback.

---

## 5. Smoke-test the real commands

```bash
# training (AUTSL + LDA) — quickest proof the pipeline works
python scripts/run_baseline.py --algo lda --lang autsl --mode real   # ~87% acc
python scripts/run_baseline.py --algo lda --lang autsl --mode both   # ~97% acc

# recording — CLI (needs webcam + display)
python scripts/record_session.py --signer piseth --lang khmer --synthetic 1 --mannequin 1
```

If both run, the migration is complete.

---

## 6. Recording via the web control panel (recommended)

Instead of the CLI flags above, drive recording from a localhost web page:

```bash
./run_web.sh              # = source venv/bin/activate && python -m webapp
```

The browser opens a control panel; the live camera + mannequin is a **separate
desktop window** (unchanged from the CLI recorder). From the page you can:

- pick a language folder or **create** a new one, and set the signer tag;
- **add labels**, **record** a take per label, **delete** one bad take or all of
  them — counts update live because they are scanned from disk, never cached;
- change **mannequin / synthetic / duration** live with sliders;
- switch between **Record** and **Recognize** modes (Recognize is a Phase-2 stub:
  no model is saved yet, so live inference is not built).

Architecture (see `webapp/`): Flask runs in a daemon thread and only flips shared
state; the OpenCV/Open3D window is pumped from the **main thread** (both toolkits
require it). `webapp/library.py` is the single source of truth for the recording
folders — the same functions the CLI already uses (`save_pair`,
`generate_variants`, `wipe_synthetic`) do the actual work.

---

## Notes
- The old Godot / WSL / UDP live-demo pipeline does NOT come to Linux and is
  deprecated — ignore any references to it.
- Re-download the raw AUTSL dataset (the 34 GB `archive/`) from Kaggle only if
  you need to re-import from scratch. The already-imported landmark data in
  `data/sequences_v2/autsl` is enough to train.
