# scripts/

Command-line tools. Run from `khmer_sign_recognizer/` with the venv active;
every one takes `--help`. Full usage: [docs/guides/COMMANDS.md](../docs/guides/COMMANDS.md).

## Recording
| | |
|---|---|
| `record_session.py` | CLI recorder. The web panel (`./run_web.sh`) is the usual way in; this is for remote machines or scripted sessions. |
| `mannequin_local.py` | 3D viewer and playback. `--playback` replays saved takes with no camera — the quickest way to confirm an import looks like the sign. |

## Before you share data
| | |
|---|---|
| `check_camera.py` | **Run this first when live recognition disappoints.** Reports how often the tracker actually finds each hand. 92% of the model's signal is the hands. |
| `check_labels.py` | Does your `labels.json` agree with the team's? A mismatch trains the wrong thing with no error. |
| `verify_pool.py` | Duplicate signer tags, missing sidecars, wrong shapes, synthetic that no longer divides evenly. Run after **every** import. |
| `export_recordings.py` | Collect your own takes into one folder for Drive. `--lang` for one corpus, and synthetic is skipped on purpose. |

## Pooling and data
| | |
|---|---|
| `import_takes.py` | Take a teammate's folder however they structured it and merge it in. Signer comes from the folder, not the filename; identical clips are deduped by content hash. |
| `generate_synthetic.py` | Rebuild each real take on differently-proportioned bodies. **Always `--clean`.** |
| `relabel.py` | Rename a label and rewrite it inside every sidecar. Nothing is deleted. |
| `import_dataset.py` | Convert an external dataset (AUTSL parquet) into our schema. |
| `drive_sync.py` | rclone wrapper. |

## Training
| | |
|---|---|
| `run_baseline.py` | Train ONE classical algorithm and report metrics. `--list` shows every algorithm, including anything in `custom_algos/`. |
| `compare_detectors.py` | MediaPipe vs RTMPose capture comparison. |

Deep models train through `python -m src.v2.train`, not from here.
