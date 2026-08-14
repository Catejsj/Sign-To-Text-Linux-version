"""Drop a teammate's folder in, get working data out.

Everyone records differently and names things differently. Rather than asking
eight people to follow a naming convention, this takes whatever they uploaded
and rewrites it into the layout the rest of the project expects.

It accepts real and synthetic, clean and noisy, any folder nesting, missing or
broken .json sidecars, and repeated imports of the same person. Nothing is ever
overwritten: incoming takes are renumbered onto the end of what is already
there.

USAGE
-----
    # the usual case — Drive folder TaskA lands in khmer_var, TaskB in khmer
    python scripts/import_takes.py ~/Downloads/TaskA
    python scripts/import_takes.py ~/Downloads/TaskB

    # look first, write nothing
    python scripts/import_takes.py ~/Downloads/TaskA --dry-run

    # override where it goes, or who it belongs to
    python scripts/import_takes.py ~/Downloads/stuff --lang khmer_var
    python scripts/import_takes.py ~/Downloads/stuff --signer dara

WHO A TAKE BELONGS TO
---------------------
Signer identity still matters — it is what makes "can this recognise someone
it has never seen?" answerable. But nobody has to type it correctly while
recording. It is taken from, in order:

    --signer  >  the folder each file sits under inside SRC  >  the filename

So if everyone drops their export into TaskA/<their name>/, it just works, and
whatever they typed into the panel is irrelevant.

Afterwards, ./run_web.sh and the training scripts see it as ordinary data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v2.schema import (                                    # noqa: E402
    SEQ_LEN, NUM_JOINTS, NUM_COORDS, SampleMeta, Source, View,
)

SEQUENCES = ROOT / "data" / "sequences_v2"
SHAPE = (SEQ_LEN, NUM_JOINTS, NUM_COORDS)

STEM = re.compile(
    r"^(?P<signer>.+)__(?P<source>[a-z]+)__(?P<view>[a-z]+)__(?P<variant>\d+)$")
SLUG = re.compile(r"^sl_\d+$")

# Drive folder name -> language folder. Anything else needs --lang.
FOLDER_LANG = {"taska": "khmer_var", "taskb": "khmer"}


def _squash(name: str) -> str:
    """Lowercase, drop everything that isn't a letter or digit."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def guess_lang(src: Path) -> str | None:
    """Which language a downloaded folder belongs to, from its name.

    Google Drive hands back names like 'TaskA-20260814T0930Z-001' and a second
    download gives 'TaskA (1)', so match on a prefix of the squashed name
    rather than the whole thing.
    """
    squashed = _squash(src.name)
    hits = {lang for key, lang in FOLDER_LANG.items()
            if squashed.startswith(key)}
    return hits.pop() if len(hits) == 1 else None


def read_sidecar(npy: Path) -> dict:
    """The .json next to a take, or {} if absent or unreadable."""
    f = npy.with_suffix(".json")
    if not f.exists():
        return {}
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}


def find_label(path: Path, src: Path, labels: dict) -> str | None:
    """Which sign this take is, from the folder above it."""
    texts = {v: k for k, v in labels.items()}
    for parent in path.parents:
        if parent == src.parent:
            break
        name = parent.name
        if name in labels or SLUG.match(name):
            return name
        if name in texts:                       # folder named in Khmer
            return texts[name]
    return None


def find_signer(path: Path, src: Path, stem_signer: str | None) -> str:
    """First folder under SRC, else whatever the filename claimed."""
    try:
        rel = path.relative_to(src)
    except ValueError:
        rel = Path(path.name)
    parts = rel.parts[:-1]
    for p in parts:
        if p in ("sequences_v2", "data") or SLUG.match(p):
            continue
        if p in FOLDER_LANG.values() or _squash(p) in FOLDER_LANG:
            continue
        return p
    return stem_signer or "unknown"


def clip_hash(arr: np.ndarray) -> str:
    """Content fingerprint of one take, used to spot re-imports."""
    return hashlib.blake2b(
        np.ascontiguousarray(arr, dtype=np.float32).tobytes(),
        digest_size=16).hexdigest()


def existing_hashes(dest_label: Path) -> set:
    """Fingerprints of every take already in a label folder, any signer."""
    out = set()
    if not dest_label.is_dir():
        return out
    for p in dest_label.glob("*.npy"):
        try:
            out.add(clip_hash(np.load(p)))
        except Exception:                                     # noqa: BLE001
            continue
    return out


def next_free(dest_label: Path, signer: str, source: str) -> int:
    n = -1
    for p in dest_label.glob(f"{signer}__{source}__*__*.npy"):
        m = STEM.match(p.stem)
        if m:
            n = max(n, int(m["variant"]))
    return n + 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="folder downloaded from the Drive")
    ap.add_argument("--lang", default=None,
                    help="destination language folder (default: from the "
                         "source folder name, TaskA/TaskB)")
    ap.add_argument("--signer", default=None,
                    help="force every take to this signer tag")
    ap.add_argument("--labels-from", default="khmer",
                    help="language to copy labels.json from if the "
                         "destination has none (default: khmer)")
    ap.add_argument("--allow-duplicates", action="store_true",
                    help="import takes even if identical ones are already "
                         "there (default: skip them, so re-importing a pool "
                         "that contains your own upload is safe)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen, write nothing")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()

    # Drive hands back a .zip -- take it as-is rather than making people unzip.
    tmp: tempfile.TemporaryDirectory | None = None
    if src.is_file() and src.suffix.lower() == ".zip":
        lang = args.lang or guess_lang(src.with_suffix(""))
        if not lang:
            sys.exit(f"cannot tell which task '{src.name}' is.\n"
                     f"Name it TaskA or TaskB, or pass --lang.")
        tmp = tempfile.TemporaryDirectory()
        print(f"unzipping {src.name} ...")
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp.name)
        src = Path(tmp.name)
    elif not src.is_dir():
        sys.exit(f"not a folder or .zip: {src}")
    else:
        lang = args.lang or guess_lang(src)
        if not lang:
            sys.exit(
                f"cannot tell which task '{src.name}' is.\n"
                f"Name the folder TaskA or TaskB, or pass "
                f"--lang khmer_var (Task A) / --lang khmer (Task B).")

    dest = SEQUENCES / lang

    # labels.json: inherit rather than make people retype Khmer
    src_labels = SEQUENCES / args.labels_from / "labels.json"
    dest_labels = dest / "labels.json"
    if not dest_labels.exists():
        if not src_labels.exists():
            sys.exit(f"no labels.json in {dest} and none at {src_labels}")
        if not args.dry_run:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_labels, dest_labels)
        print(f"labels.json copied from '{args.labels_from}' into '{lang}'")
    labels = json.loads(
        (dest_labels if dest_labels.exists() else src_labels)
        .read_text(encoding="utf-8"))

    # ---- collect: (label, signer, source, old_variant) -> {view: path}
    #
    # The .json sidecar is authoritative -- we wrote it, and it survives people
    # renaming files. The filename is the fallback for takes that arrive
    # without one. Only the signer is decided elsewhere (the upload folder),
    # so nobody has to have typed their tag correctly.
    SOURCES = {s.value for s in Source}
    VIEWS = {v.value for v in View}

    takes: dict = defaultdict(dict)
    skipped: list[str] = []
    unlabelled = 0
    for npy in sorted(src.rglob("*.npy")):
        side = read_sidecar(npy)
        m = STEM.match(npy.stem)

        label = (side.get("label") if side.get("label") in labels
                 else None) or find_label(npy, src, labels)
        if not label:
            unlabelled += 1
            skipped.append(f"{npy.relative_to(src)}: cannot tell which sign "
                           f"(no label folder above it and no .json sidecar)")
            continue

        source = next((v for v in (side.get("source"),
                                   m["source"] if m else None)
                       if v in SOURCES), "real")
        view = next((v for v in (side.get("view"), m["view"] if m else None)
                     if v in VIEWS), "clean")

        old = side.get("variant")
        if not isinstance(old, int):
            old = int(m["variant"]) if m else len(takes)

        signer = args.signer or find_signer(
            npy, src, side.get("signer_id") or (m["signer"] if m else None))
        takes[(label, signer, source, old)][view] = npy

    if not takes:
        print("Found no importable .npy files.")
        for s in skipped[:10]:
            print(f"  skipped {s}")
        return

    # ---- renumber onto the end of what is already there
    groups: dict = defaultdict(list)
    for (label, signer, source, old), views in takes.items():
        groups[(label, signer, source)].append((old, views))

    written = 0
    duplicates = 0
    per_signer: dict = defaultdict(lambda: defaultdict(int))
    written_grid: dict = defaultdict(int)
    warnings: list[str] = []
    seen_cache: dict = {}

    for (label, signer, source), entries in sorted(groups.items()):
        dest_label = dest / label
        start = next_free(dest_label, signer, source) \
            if dest_label.exists() else 0
        # Content hashes of everything already in this label folder, under ANY
        # signer tag. Re-importing a pool that contains your own upload would
        # otherwise duplicate your takes -- and because the tag comes from the
        # Drive folder, the copies can land under a different spelling of your
        # name, which reads as a second person and leaks across a
        # leave-one-signer-out split.
        if label not in seen_cache:
            seen_cache[label] = existing_hashes(dest_label)
        seen = seen_cache[label]

        pending = []
        for _old, views in sorted(entries):
            clips: dict = {}
            for view, npy in sorted(views.items()):
                try:
                    clip = np.load(npy).astype(np.float32)
                except Exception as exc:                      # noqa: BLE001
                    warnings.append(f"{npy.name} will not load: {exc}")
                    continue
                if clip.shape != SHAPE:
                    warnings.append(
                        f"{npy.name} has shape {clip.shape}, expected {SHAPE} "
                        f"— skipped")
                    continue
                if not np.isfinite(clip).all():
                    clip = np.nan_to_num(clip, nan=0.0, posinf=0.0, neginf=0.0)
                    warnings.append(f"{npy.name} had NaN/inf — zeroed")
                clips[view] = clip
            if not clips:
                continue
            digests = {v: clip_hash(c) for v, c in clips.items()}
            if not args.allow_duplicates and all(d in seen
                                                 for d in digests.values()):
                duplicates += 1
                continue
            pending.append((clips, digests))

        for i, (clips, digests) in enumerate(pending):
            variant = start + i
            for view, clip in sorted(clips.items()):
                stem = f"{signer}__{source}__{view}__{variant:04d}"
                if not args.dry_run:
                    dest_label.mkdir(parents=True, exist_ok=True)
                    np.save(dest_label / f"{stem}.npy", clip)
                    meta = SampleMeta(
                        label=label, signer_id=signer, source=Source(source),
                        view=View(view), language=lang, variant=variant,
                        notes=f"imported from {src.name}")
                    (dest_label / f"{stem}.json").write_text(
                        meta.to_json(), encoding="utf-8")
                seen.add(digests[view])
                written += 1
            per_signer[signer][source] += 1
            if source == "real":
                written_grid[(signer, label)] += 1

    # ---- report
    print(f"\n{'would import' if args.dry_run else 'imported'} into "
          f"data/sequences_v2/{lang}/")

    # real takes per person per sign — the table that shows who did what
    grid = written_grid
    people = sorted({s for s, _ in grid})
    signs = sorted({l for _, l in grid})
    if people and signs:
        w = max(len(p) for p in people) + 2
        print("\n  real takes added, per person")
        print("  " + "person".ljust(w) + "".join(s.rjust(9) for s in signs)
              + "total".rjust(9))
        for p in people:
            row = "".join(str(grid.get((p, s), 0)).rjust(9) for s in signs)
            tot = sum(grid.get((p, s), 0) for s in signs)
            print("  " + p.ljust(w) + row + str(tot).rjust(9))

    print()
    for signer in sorted(per_signer):
        parts = ", ".join(f"{n} {s}" for s, n in
                          sorted(per_signer[signer].items()))
        print(f"  {signer:<15} {parts}")
    print(f"  {'':<15} {written} files total")
    if duplicates:
        print(f"\n  skipped {duplicates} take(s) already present "
              f"(same content) — re-import is safe")

    ratios: dict = defaultdict(lambda: {"real": 0, "synthetic": 0})
    for (label, signer, source), entries in groups.items():
        if source in ("real", "synthetic"):
            ratios[(label, signer)][source] += len(entries)
    for (label, signer), c in sorted(ratios.items()):
        if c["synthetic"] and c["real"] and c["synthetic"] % c["real"]:
            warnings.append(
                f"{label}/{signer}: {c['synthetic']} synthetic against "
                f"{c['real']} real does not divide evenly — regenerate "
                f"synthetic for this language instead of trusting it")

    if skipped:
        print(f"\nskipped {len(skipped)} file(s) with no label folder:")
        for s in skipped[:5]:
            print(f"  {s}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")
    if warnings:
        print("\nwarnings:")
        for w in warnings[:10]:
            print(f"  {w}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")

    if not args.dry_run:
        print(f"\nDone. ./run_web.sh will show '{lang}' in the language "
              f"dropdown, and training can use --lang {lang}.")


if __name__ == "__main__":
    main()
