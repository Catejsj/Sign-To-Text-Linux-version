# ruff: noqa
"""Colab training entrypoint for the v2 Transformer.

HOW TO USE IN COLAB
-------------------
Open a fresh Colab notebook and paste EACH block below into its own cell.
Or: open this file directly from the GitHub repo in Colab
    (File → Open notebook → GitHub → Catejsj/Sign-to-Text → this file).

The notebook:
    1. mounts your student Google Drive
    2. clones the repo (or pulls latest)
    3. copies data from Drive → /content
    4. trains on the GPU
    5. copies weights back to Drive

Drive layout expected:
    /MyDrive/SignLink/data/sequences_v2/<label>/<signer>__<source>__NNNN.{npy,json}
    /MyDrive/SignLink/models/weights_v2/          ← trained weights land here
    /MyDrive/SignLink/logs/v2/                    ← training logs land here
"""

# ============================================================================
# CELL 1 — mount Drive
# ============================================================================
# from google.colab import drive
# drive.mount('/content/drive')

# ============================================================================
# CELL 2 — clone or update the repo
# ============================================================================
# !git config --global --add safe.directory /content/Sign-to-Text
# import os
# if not os.path.isdir('/content/Sign-to-Text'):
#     !git clone https://github.com/Catejsj/Sign-to-Text.git /content/Sign-to-Text
# else:
#     !cd /content/Sign-to-Text && git fetch && git reset --hard origin/main
# %cd /content/Sign-to-Text/khmer_sign_recognizer

# ============================================================================
# CELL 3 — install deps (torch is pre-installed in Colab)
# ============================================================================
# !pip -q install numpy

# ============================================================================
# CELL 4 — sync data Drive → /content
# ============================================================================
# !mkdir -p data/sequences_v2 models/weights_v2 logs/v2
# !rsync -a --delete \
#     /content/drive/MyDrive/SignLink/data/sequences_v2/ \
#     data/sequences_v2/

# ============================================================================
# CELL 5 — train
# ============================================================================
# from src.v2.train import TrainConfig, train
# cfg = TrainConfig(
#     data_root="data/sequences_v2",
#     weights_out="models/weights_v2/ksl_transformer_latest.pt",
#     labels_out="models/weights_v2/labels.json",
#     batch_size=32,
#     epochs=100,
#     lr=1e-3,
#     held_out_signer=None,    # e.g. "Menghong" for leave-one-signer-out
#     model_type="tcn",        # "tcn" (default, recommended) or "transformer"
# )
# result = train(cfg)
# print("best val acc:", result["best_val_acc"])

# ============================================================================
# CELL 6 — push weights + log back to Drive
# ============================================================================
# import json, shutil, datetime
# run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# dst_weights = f"/content/drive/MyDrive/SignLink/models/weights_v2/"
# dst_logs    = f"/content/drive/MyDrive/SignLink/logs/v2/"
# !mkdir -p "{dst_weights}" "{dst_logs}"
# !cp models/weights_v2/ksl_transformer_latest.pt \
#    "{dst_weights}/ksl_transformer_{run_id}.pt"
# !cp models/weights_v2/ksl_transformer_latest.pt \
#    "{dst_weights}/ksl_transformer_latest.pt"
# !cp models/weights_v2/labels.json "{dst_weights}/labels.json"
# with open(f"/content/Sign-to-Text/khmer_sign_recognizer/logs/v2/{run_id}.json", "w") as f:
#     json.dump(result, f, indent=2, default=str)
# !cp logs/v2/{run_id}.json "{dst_logs}/"
# print("done. run_id:", run_id)


if __name__ == "__main__":
    # Allows running locally for sanity: `python notebooks/colab_train_v2.py`.
    # Assumes you've already placed samples under data/sequences_v2/.
    from pathlib import Path
    import sys
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from src.v2.train import TrainConfig, train

    cfg = TrainConfig(
        data_root=str(ROOT / "data" / "sequences_v2"),
        weights_out=str(ROOT / "models" / "weights_v2" / "ksl_transformer_latest.pt"),
        labels_out=str(ROOT / "models" / "weights_v2" / "labels.json"),
        epochs=20,
        device="cuda",
    )
    train(cfg)
