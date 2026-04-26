# SignLink — Khmer Sign Language Recognizer

Proof-of-concept sign-to-text model for Khmer Sign Language (KSL).
Year-2 research project at CamTech University.

## Where to start

| If you are… | Read |
|---|---|
| a new team member | [`WORKFLOW.md`](WORKFLOW.md) |
| setting up training on Colab | [`WORKFLOW.md`](WORKFLOW.md) → "Colab bootstrap" |
| working on the CNN track | [`WORKFLOW.md`](WORKFLOW.md) → "Teammate workflow" + [`notebooks/train_ksl.py`](notebooks/train_ksl.py) |
| curious about the architecture | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| setting up the live Godot demo | [`docs/SETUP_legacy.md`](docs/SETUP_legacy.md) |

## Two tracks in one repo

- **v2 (main)** — `src/v2/`, Transformer encoder on 48-joint skeleton sequences. Owned by the leads.
- **legacy CNN** — `src/model.py`, `src/dataset.py`, `notebooks/train_ksl.py`. Open for any teammate to iterate on.

Both share the camera capture stack in [`src/capture.py`](src/capture.py) and the Godot mannequin at [`khmer-sign-mannequin2/`](khmer-sign-mannequin2/).

## Repo layout

```
khmer_sign_recognizer/
├── README.md
├── WORKFLOW.md                     team map + Colab + Drive workflow
├── requirements.txt                runtime deps (Windows capture)
├── requirements-train.txt          training deps (Colab)
├── config/settings.json            IPs, capture params, filter tuning
├── src/
│   ├── capture.py                  RTMPose + MediaPipe capture
│   ├── mapper.py, main_wsl.py      WSL pose→bone bridge (Pipeline A)
│   ├── bridge.py, send_to_wsl.py   UDP plumbing
│   ├── model.py, dataset.py        legacy CNN track
│   └── v2/                         new Transformer track
├── scripts/
│   ├── record_session.py           v2 streaming recorder (paired clean+noisy)
│   └── drive_sync.py               rclone wrapper
├── notebooks/
│   ├── train_ksl.py                legacy CNN Colab script
│   └── colab_train_v2.py           v2 Transformer Colab script
├── khmer-sign-mannequin2/          Godot 4.6 project (Y-Bot)
├── data/, models/, logs/           gitignored — synced via Drive
├── docs/                           architecture + legacy setup
├── run_windows.py, start_*.sh      Pipeline A launchers
├── update_ips.py                   refresh WSL IPs in settings.json
├── record_signs.py                 legacy recorder (CNN track)
└── inference.py                    legacy real-time inference
```

## Quick commands

```powershell
# Pipeline A live demo (Windows + WSL + Godot)
python update_ips.py                         # refresh WSL IPs
wsl -e bash -c "cd /mnt/d/Projects/Sign\ to\ Text/khmer_sign_recognizer && ./start_wsl.sh"
start_windows.bat                            # Windows side
# open khmer-sign-mannequin2/ in Godot and F5

# v2 recording + training cycle
python scripts/record_session.py --signer <your-name>   # streaming session
python scripts/drive_sync.py push-data
# → open notebooks/colab_train_v2.py in Colab, run all cells
python scripts/drive_sync.py pull-weights
```

See [`WORKFLOW.md`](WORKFLOW.md) for the full cycle.
