# SignLink on macOS

Works on both Intel and Apple Silicon. **Everything runs on the CPU** — there is
no CUDA on a Mac, and RTMPose has no Apple-GPU path, so pose tracking is roughly
10× slower than on an NVIDIA machine. That is fine for recording takes; live
Recognize mode will feel sluggish.

## Setup

```bash
git clone https://github.com/Catejsj/Sign-To-Text-Linux-version.git
cd Sign-To-Text-Linux-version/khmer_sign_recognizer

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` picks the right ONNX runtime for your platform
automatically — macOS gets the CPU build, Linux and Windows get the GPU one.
There is nothing to edit.

## Running

```bash
./run_web.sh
```

Then open **http://127.0.0.1:8000**.

The camera and mannequin appear in a **separate desktop window**, not in the
browser. The browser page is the control panel only.

## The two things that catch people out

**Do not run `scripts/record_session.py`.** It also opens a browser page, on a
random port, and looks like an older version of the app — because it is. The
only launch command is `./run_web.sh`.

**Camera permission.** The first time you record, macOS asks the app running
Python (Terminal, iTerm, or VS Code) for camera access. If it is denied, the
camera silently fails to open and the log shows "Failed to open camera". Fix it
in System Settings → Privacy & Security → Camera, then restart the terminal —
the permission only applies to newly launched processes.

## Troubleshooting

**`permission denied: ./run_web.sh`**
The file lost its executable bit. Either is fine:
```bash
bash run_web.sh          # works regardless
chmod +x run_web.sh      # fixes it permanently
```

**`No matching distribution found for onnxruntime-gpu`**
You are on an old clone from before the platform split. `git pull`, then
reinstall.

**Open3D fails to install or import**
The mannequin is optional. The panel detects its absence and runs camera-only;
nothing else breaks.

**"No CUDA provider — RTMPose will run on CPU"**
Expected on a Mac. Not an error.

**Blank/black camera window**
Usually the permission issue above. If permission is granted and it is still
black, try `camera_id: 1` in `config/settings.json` — Macs with Continuity
Camera sometimes enumerate the iPhone as camera 0.

## What is different from Linux

| | Linux (NVIDIA) | macOS |
|---|---|---|
| Pose backend | CUDA, ~40 fps | CPU, ~4 fps |
| ONNX runtime | `onnxruntime-gpu` | `onnxruntime` |
| Camera backend | V4L2 | AVFoundation (auto) |
| Recording takes | yes | yes |
| Live recognition | yes | slow, not recommended |

Recording is the part that matters for contributing data, and it works. See
`docs/TEAM_DATA_COLLECTION_PLAN.md` for what to record and how to upload it.
