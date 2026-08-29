# Windows setup (for teammates)

The project is cross-platform: every OS-specific branch is selected at runtime
(camera backend, GPU library loading, 3D window handling). Windows teammates and
Linux teammates share the same code and the same recorded data — only the
install steps differ.

---

## 1. Install Python 3.11

**Not 3.12+.** `mediapipe` 0.10.14 and `open3d` publish no wheels beyond
Python 3.12, so a newer interpreter cannot install the stack at all.

Get it from [python.org](https://www.python.org/downloads/release/python-3119/)
and tick **"Add python.exe to PATH"**. Check:

```cmd
py -3.11 --version
```

## 2. Clone and create the virtual environment

```cmd
git clone <repo-url>
cd khmer_sign_recognizer
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

## 3. Install dependencies

Install **torch first** — the CUDA build must match `onnxruntime-gpu`, which is
built against CUDA 12. The default PyPI torch now ships CUDA 13 and will not
work with it.

```cmd
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**Then repair onnxruntime.** `rtmlib` depends on the CPU `onnxruntime` package,
which installs *over* `onnxruntime-gpu` and silently removes GPU support:

```cmd
pip list | findstr onnxruntime
pip uninstall -y onnxruntime
pip install --force-reinstall onnxruntime-gpu==1.24.4
```

A later warning that `rtmlib requires onnxruntime` is harmless —
`onnxruntime-gpu` provides the same module.

No CUDA toolkit install is needed; the CUDA libraries ship inside the torch
wheel. Only an NVIDIA driver is required. Without an NVIDIA GPU everything still
runs, just slower on CPU.

## 4. Verify

```cmd
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import sys; sys.path.insert(0,'.'); from src.cuda_setup import assert_onnx_gpu; assert_onnx_gpu(); print('ONNX GPU OK')"
python -c "import cv2, mediapipe, sklearn, open3d, rtmlib; print('imports OK')"
```

`assert_onnx_gpu()` matters: `onnxruntime.get_available_providers()` lists
providers the build supports, **not** those that can actually load. ONNX Runtime
falls back to CPU silently, so it must be verified by building a real session.

## 5. Run

```cmd
run_web.bat
```

Opens the control panel in a browser. The live camera + mannequin appears as a
separate desktop window. Command-line equivalents also work:

```cmd
python -m webapp
python scripts\run_baseline.py --algo rf --lang khmer --mode both --split-random 0.2
```

---

## Platform differences (handled automatically)

| area | Windows | Linux |
|---|---|---|
| Camera backend | `CAP_DSHOW` | `CAP_V4L2` |
| GPU library loading | `os.add_dll_directory` | `ctypes` `RTLD_GLOBAL` preload |
| Open3D windowing | native | forced to X11 when on Wayland |
| Launcher | `run_web.bat` | `run_web.sh` |

Nothing needs configuring — these are selected from `sys.platform` at runtime.

---

## Sharing recordings

Recorded data is **git-ignored** (`data/`, `*.npy`) because it is large and
binary. Only `labels.json` is committed, so everyone's label list stays in sync.

Share the takes themselves out-of-band (Drive, shared disk) and drop them into:

```
data/sequences_v2/<language>/<label_slug>/
```

Filenames encode everything needed:
`{signer}__{real|synthetic}__{clean|noisy}__{NNNN}.npy` plus a `.json` sidecar.

**Use a unique signer tag per person.** It is what makes leave-one-signer-out
evaluation (`--holdout <name>`) possible, and that is the only honest measure of
how the system performs for someone it has not seen.

## Line endings

`.gitattributes` normalizes text files to LF in the repository and keeps `.bat`
files CRLF, so Windows and Linux teammates do not see whole files as modified.
Nothing to configure.

## Common problems

| symptom | cause and fix |
|---|---|
| `mediapipe` won't install | Python is 3.12+; use 3.11 |
| `CUDA: False` | torch installed from the default index; reinstall from the cu128 index |
| `assert_onnx_gpu()` fails | the CPU `onnxruntime` overwrote `onnxruntime-gpu`; see step 3 |
| Camera won't open | another app holds it, or a privacy shutter is closed |
| Whole files show as changed in git | `.gitattributes` not applied — re-clone, do not disable `core.autocrlf` |
