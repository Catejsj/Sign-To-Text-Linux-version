# src/

The library. Scripts and the web app import from here; nothing here is meant to
be run directly except `src.v2.train`.

## src/ — shared capture layer
| | |
|---|---|
| `capture.py` | Camera + MediaPipe + RTMPose. Picks the camera backend and GPU provider per platform. Attaches a per-joint confidence to every landmark. |
| `cuda_setup.py` | Preloads the CUDA libraries ONNX Runtime needs. Without it ONNX silently runs on CPU — see PROBLEM_LOG §A3. |
| `utils.py` | Config loading and logging. |

**`src/v2/` sits on top of these — it does not replace them.**

## src/v2/ — the pipeline
Read in this order:

| | |
|---|---|
| `schema.py` | **The data contract.** Filenames, `(60, 48, 3)`, joint layout. Change shapes here or nowhere. |
| `normalize.py` | Raw landmarks → clean/noisy views. Resample, fill gaps, shoulder-normalise, de-roll. |
| `landmarks.py` | Which hands were really seen. `fill_nans` erases the evidence of tracking failure; this recovers it. Worth +15 macro-F1 on an unseen signer (PROBLEM_LOG §K). |
| `bones.py` | Parent-relative unit vectors — the only thing telling a model the joints form a skeleton. Directions, not lengths: length is body size, i.e. signer identity (§M). |
| `retarget.py` | Synthetic signers by skeletal retargeting. |
| `augment.py` | Time-warp, noise, rotation. |
| `dataset.py` | Sample discovery and **take-aware** splitting. `split_random` holds out takes, not samples — splitting by sample leaked 100% (§C4). |
| `baseline_data.py` | Feature extraction for classical models. `bones` is the default. |
| `baseline_eval.py` | Metrics and reports. |
| `algorithms.py` | The algorithm registry, and the `custom_algos/` loader. |
| `model_tcn.py`, `model_rnn.py`, `model_transformer.py` | The six architectures. |
| `train.py` | Deep training loop. `python -m src.v2.train --lang khmer --model gru` |
| `recognizer.py` | Saved-model bundles and live prediction. A bundle records how its input was built, so inference rebuilds it identically. |

## Two invariants worth knowing before editing

**Take-aware splitting.** A take writes a clean view, a noisy view and several
synthetic copies. They must never straddle a split. Everything routes through
`dataset.take_id()`.

**Presence before augmentation.** `landmarks.hand_presence` reads the raw clip;
augmentation noise would destroy the signature it detects. Bones are computed
*after* augmentation, because they must describe the clip the model sees.
