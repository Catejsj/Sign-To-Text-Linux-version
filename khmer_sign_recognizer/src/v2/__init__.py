"""
v2 — redesigned training pipeline.

Contract:
  - sample shape: (60, 48, 3) float32
  - joint layout: src.v2.schema
  - 6 body (RTMPose) + 21 left hand + 21 right hand (MediaPipe)

Legacy (src/model.py, src/normalizer.py, src/dataset.py) is untouched
and continues to run. v2 coexists; it does not replace.
"""
