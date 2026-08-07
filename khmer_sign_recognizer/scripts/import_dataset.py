"""Convert an external isolated-SLR dataset into the v2 schema.

Output files match exactly what record_session.py produces, so the
existing dataset.py loader, retarget.py synthetic generator, and training
code work on imported data with zero changes.

Supported formats
-----------------
autsl
    Kaggle "AUTSL processed MediaPipe landmarks" layout. One parquet per
    sample, GISLR competition format (frame/type/landmark_index/x/y/z).
    Source folder layout:
        <src>/train.csv  val.csv  test.csv
        <src>/{train,val,test}/<participant_id>/<sequence_id>.parquet
    Needs data/external/SignList_ClassId_TR_EN.csv (already vendored).

Example
-------
    python scripts/import_dataset.py \
        --format autsl \
        --src "D:/Projects/Sign to Text/archive/AUSTL_processed_landmark" \
        --signs 8,14,20,42,65,86,93,100,173,196 \
        --lang autsl

Pass --limit N for a quick smoke test on a handful of samples per split.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.v2.schema import SEQ_LEN, NUM_JOINTS, NUM_COORDS, save_pair
from src.v2.normalize import fill_nans, resample_time, shoulder_normalize, deroll


# MediaPipe Holistic pose landmark indices that map to our 6 body slots.
# (33-joint MediaPipe pose → our shoulder/elbow/wrist subset.)
POSE_TO_BODY = {
    11: 0,  # left shoulder
    12: 1,  # right shoulder
    13: 2,  # left elbow
    14: 3,  # right elbow
    15: 4,  # left wrist
    16: 5,  # right wrist
}


def parquet_to_clip(parquet_path: Path) -> np.ndarray:
    """Load one GISLR parquet → (T, 48, 3) clip in raw [0,1] image space.

    Missing landmarks (frames where MediaPipe didn't detect a hand) stay
    as NaN; downstream `fill_nans` carries forward the last valid value.
    """
    df = pd.read_parquet(parquet_path)

    frame_ids = np.sort(df["frame"].unique())
    T = int(len(frame_ids))
    if T == 0:
        return np.zeros((1, NUM_JOINTS, NUM_COORDS), dtype=np.float32)

    frame_to_idx = {int(f): i for i, f in enumerate(frame_ids)}
    clip = np.full((T, NUM_JOINTS, NUM_COORDS), np.nan, dtype=np.float32)

    df = df.copy()
    df["t"] = df["frame"].map(frame_to_idx).astype(int)

    pose = df[(df["type"] == "pose") &
              df["landmark_index"].isin(POSE_TO_BODY)]
    if len(pose):
        j = pose["landmark_index"].map(POSE_TO_BODY).astype(int).to_numpy()
        t = pose["t"].to_numpy()
        clip[t, j, 0] = pose["x"].to_numpy()
        clip[t, j, 1] = pose["y"].to_numpy()
        clip[t, j, 2] = pose["z"].to_numpy()

    lh = df[df["type"] == "left_hand"]
    if len(lh):
        j = 6 + lh["landmark_index"].astype(int).to_numpy()
        t = lh["t"].to_numpy()
        clip[t, j, 0] = lh["x"].to_numpy()
        clip[t, j, 1] = lh["y"].to_numpy()
        clip[t, j, 2] = lh["z"].to_numpy()

    rh = df[df["type"] == "right_hand"]
    if len(rh):
        j = 27 + rh["landmark_index"].astype(int).to_numpy()
        t = rh["t"].to_numpy()
        clip[t, j, 0] = rh["x"].to_numpy()
        clip[t, j, 1] = rh["y"].to_numpy()
        clip[t, j, 2] = rh["z"].to_numpy()

    return clip


def slugify_tr(word: str) -> str:
    """The SignList CSV already transliterates Turkish to ASCII
    (e.g. "teşekkür" → "tesekkur"), so we just lowercase and tidy."""
    return word.strip().lower().replace(" ", "_")


def load_signlist() -> tuple[dict[int, str], dict[int, str]]:
    path = ROOT / "data" / "external" / "SignList_ClassId_TR_EN.csv"
    if not path.exists():
        sys.exit(
            f"missing AUTSL SignList CSV at {path}\n"
            f"download from https://data.chalearnlap.cvc.uab.cat/AuTSL/data/"
            f"SignList_ClassId_TR_EN.csv"
        )
    df = pd.read_csv(path)
    return (
        dict(zip(df["ClassId"].astype(int), df["TR"])),
        dict(zip(df["ClassId"].astype(int), df["EN"])),
    )


def import_autsl(args: argparse.Namespace) -> None:
    src = Path(args.src)
    if not src.exists():
        sys.exit(f"--src does not exist: {src}")

    id_to_tr, id_to_en = load_signlist()
    sign_ids = [int(s) for s in args.signs.split(",") if s.strip()]
    unknown = [s for s in sign_ids if s not in id_to_tr]
    if unknown:
        sys.exit(f"unknown AUTSL sign ids: {unknown}")

    print(f"importing {len(sign_ids)} signs into language='{args.lang}':")
    for sid in sign_ids:
        print(f"  {sid:>3}  {id_to_tr[sid]:<20s} ({id_to_en[sid]})")

    dst_root = Path(args.dst)
    lang_dir = dst_root / args.lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    # Display label == the slug itself (the plain Turkish word). This is
    # what teammates must TYPE in the recorder so their takes merge into
    # the same class instead of spawning a duplicate like `aile_1`. The
    # English meaning lives in data/external/SignList_ClassId_TR_EN.csv.
    labels = {
        slugify_tr(id_to_tr[sid]): slugify_tr(id_to_tr[sid])
        for sid in sign_ids
    }
    labels_path = lang_dir / "labels.json"
    labels_path.write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {labels_path}")

    total_saved = 0
    total_skipped = 0
    t0 = time.time()
    for split in ("train", "val", "test"):
        csv_path = src / f"{split}.csv"
        if not csv_path.exists():
            print(f"\n[{split}] csv missing, skip")
            continue

        df = pd.read_csv(csv_path)
        df = df[df["sign"].isin(sign_ids)].reset_index(drop=True)
        if args.limit and len(df) > args.limit:
            df = df.head(args.limit)
        print(f"\n[{split}] processing {len(df)} samples")

        for i, row in enumerate(df.itertuples(index=False)):
            parquet_path = src / row.path
            if not parquet_path.exists():
                total_skipped += 1
                continue

            try:
                raw = parquet_to_clip(parquet_path)
            except Exception as e:  # corrupted parquet, log and continue
                print(f"  [warn] {row.path}: {e}")
                total_skipped += 1
                continue

            noisy = fill_nans(resample_time(raw, SEQ_LEN))
            clean = deroll(shoulder_normalize(noisy))

            sign_id = int(row.sign)
            slug = slugify_tr(id_to_tr[sign_id])
            signer_id = f"autsl_p{int(row.participant_id):03d}_{split}"

            save_pair(
                dst_root,
                clean.astype(np.float32),
                noisy.astype(np.float32),
                label=slug,
                signer_id=signer_id,
                fps=30,
                language=args.lang,
                notes=f"autsl class_id={sign_id} split={split} "
                      f"src={row.path}",
            )
            total_saved += 1

            if (i + 1) % 50 == 0 or (i + 1) == len(df):
                elapsed = time.time() - t0
                print(f"  {i + 1}/{len(df)}  ({elapsed:.1f}s elapsed)")

    elapsed = time.time() - t0
    print(
        f"\ndone in {elapsed:.1f}s. "
        f"saved {total_saved} clean+noisy pairs, skipped {total_skipped}."
    )
    print(f"data lives at {lang_dir}/")
    print("next steps:")
    print(f"  python scripts/generate_synthetic.py --language {args.lang} "
          f"--per-take 6")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Import an external SLR dataset into the v2 schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--format", required=True, choices=["autsl"],
                    help="source dataset layout (only 'autsl' for now)")
    ap.add_argument("--src", required=True,
                    help="path to the dataset root folder")
    ap.add_argument("--signs", required=True,
                    help="comma-separated class ids to import "
                         "(e.g. 8,14,20,42,65,86,93,100,173,196)")
    ap.add_argument("--lang", default="autsl",
                    help="target language folder name (default: autsl)")
    ap.add_argument("--dst", default=str(ROOT / "data" / "sequences_v2"),
                    help="target root (default: data/sequences_v2)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap per split (0 = no cap). Useful for smoke tests.")
    args = ap.parse_args()

    if args.format == "autsl":
        import_autsl(args)


if __name__ == "__main__":
    main()
