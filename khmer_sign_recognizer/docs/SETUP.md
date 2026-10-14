# Setup

How to get SignLink running on Linux, Windows or macOS. The code is the same
everywhere; only the install steps differ. Every OS-specific choice (camera
backend, GPU library loading, 3D window) is picked automatically at runtime.

---

## The two rules that break most installs

**1. Python 3.11. Not 3.12, not 3.14.**
`mediapipe` 0.10.14 and `open3d` publish no wheels past Python 3.12, so a newer
interpreter cannot install the stack at all. CachyOS ships 3.14 as `python`, so
on Linux you must name `python3.11` explicitly.

**2. Install torch first, from the CUDA 12 index.**
The default PyPI torch now ships CUDA 13. `onnxruntime-gpu`, which runs the
RTMPose body tracker, is built against CUDA 12. If they mismatch, the tracker
silently falls back to the CPU and everything runs about ten times slower with
no error.

You do **not** need a system CUDA toolkit. The CUDA libraries come inside the
torch wheel; only an NVIDIA driver is required. Without an NVIDIA GPU
everything still works, just slower.

---

## Linux (CachyOS / Arch)

```bash
sudo pacman -S --needed git base-devel noto-fonts python311
nvidia-smi                                   # driver check: should list the GPU

git clone https://github.com/Catejsj/Sign-To-Text-Linux-version.git
cd Sign-To-Text-Linux-version/khmer_sign_recognizer

python3.11 -m venv venv                      # NOT `python -m venv`, that is 3.14
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-linux.txt
```

Then do the onnxruntime repair below. Do **not** `pacman -S cuda cudnn`, and do
**not** pin `numpy==1.26.4` if an old note tells you to: `opencv-python` 4.13
requires numpy 2, so that pin makes the install unresolvable.

If git complains about "dubious ownership" after copying the folder from
another drive:

```bash
git config --global --add safe.directory "$(pwd)/.."
```

Known-good versions are frozen in `requirements-lock-linux.txt`.

## Windows

Install Python 3.11 from python.org and tick **Add python.exe to PATH**.

```cmd
git clone https://github.com/Catejsj/Sign-To-Text-Linux-version.git
cd Sign-To-Text-Linux-version\khmer_sign_recognizer

py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, run this once in an admin
PowerShell:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then do the onnxruntime repair below. Known-good versions are frozen in
`requirements-lock-windows.txt`.

## macOS

Works on Intel and Apple Silicon, **CPU only**: there is no CUDA on a Mac, so
pose tracking runs at roughly 4 fps instead of 40. Recording works fine; live
recognition will feel sluggish.

```bash
git clone https://github.com/Catejsj/Sign-To-Text-Linux-version.git
cd Sign-To-Text-Linux-version/khmer_sign_recognizer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` picks the CPU ONNX runtime on macOS automatically. Skip the
onnxruntime repair.

The first time you record, macOS asks Terminal (or iTerm, or VS Code) for
camera access. If it was denied, the camera fails silently. Fix it in
**System Settings → Privacy & Security → Camera**, then restart the terminal.

---

## The onnxruntime repair (Linux and Windows)

`rtmlib` depends on the **CPU** `onnxruntime` package, and pip installs it on
top of `onnxruntime-gpu` in the same directory, which strips GPU support out
entirely. After any `pip install -r ...`, check:

```bash
pip list | grep onnxruntime          # Windows: pip list | findstr onnxruntime
```

If plain `onnxruntime` is listed:

```bash
pip uninstall -y onnxruntime
pip install --force-reinstall "onnxruntime-gpu==1.24.4"
```

A warning afterwards that `rtmlib requires onnxruntime` is harmless.
`onnxruntime-gpu` provides the same module; pip just cannot tell.

---

## Verify

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import sys; sys.path.insert(0,'.'); from src.cuda_setup import assert_onnx_gpu; assert_onnx_gpu(); print('ONNX GPU OK')"
python -c "import cv2, mediapipe, sklearn, open3d, rtmlib, torchvision; print('imports OK')"
```

The middle line is the one that matters. `onnxruntime.get_available_providers()`
lists what the build *supports*, not what can actually *load*, and ONNX Runtime
falls back to the CPU without raising. `assert_onnx_gpu()` builds a real session
and fails loudly if the GPU is not really in use. (On macOS, skip it.)

**Already installed before torchvision was added?** Add it from the same index
as torch, so the versions match:

```bash
pip install torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Run

```bash
./run_web.sh          # Windows: run_web.bat
```

A control panel opens at **http://127.0.0.1:8000**. The camera and 3D mannequin
open in a **separate desktop window**, not in the browser. The browser page is
controls only.

Next: [USAGE.md](USAGE.md) for recording, training and evaluation.

---

## Platform differences (all automatic)

| | Linux | Windows | macOS |
|---|---|---|---|
| Camera backend | V4L2 | DirectShow | AVFoundation |
| GPU library loading | `ctypes` preload with `RTLD_GLOBAL` | `os.add_dll_directory` | none (CPU) |
| ONNX runtime | `onnxruntime-gpu` | `onnxruntime-gpu` | `onnxruntime` |
| 3D window | forced to X11 under Wayland | native | native |
| Launcher | `run_web.sh` | `run_web.bat` | `run_web.sh` |

The GPU loading lives in `src/cuda_setup.py`. On Linux it dlopens the CUDA
libraries from `site-packages/nvidia/*/lib` before the first ONNX session;
`LD_LIBRARY_PATH` cannot do this from inside Python because glibc reads it once
at process start.

`.gitattributes` keeps text files LF in the repository and `.bat` files CRLF,
so Windows and Linux teammates do not see whole files as modified.

---

## Common problems

| symptom | cause and fix |
|---|---|
| `mediapipe` will not install | Python is 3.12 or newer. Rebuild the venv with 3.11 |
| `CUDA: False` | torch came from the default index. Reinstall from the cu128 index |
| `assert_onnx_gpu()` fails | the CPU `onnxruntime` overwrote the GPU one. Do the repair above |
| camera will not open | another app holds it (Zoom, Teams, a browser tab), or a privacy shutter is closed |
| camera window is black on macOS | camera permission (see macOS above). With Continuity Camera, try `camera_id: 1` in `config/settings.json` |
| no 3D window on Linux | Open3D on Wayland. The code forces X11 automatically; if it still fails: `env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 python -m webapp` |
| `permission denied: ./run_web.sh` | `chmod +x run_web.sh`, or run `bash run_web.sh` |
| whole files show as changed in git | `.gitattributes` not applied. Re-clone; do not disable `core.autocrlf` |
| Open3D fails to install | the mannequin is optional. The panel runs camera-only without it |
