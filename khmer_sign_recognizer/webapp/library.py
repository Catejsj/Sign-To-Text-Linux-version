"""Filesystem source-of-truth for the recording library.

Every count returned here is derived by **scanning the folder**, never stored.
That is the whole point: deleting takes — from this app or from the file
explorer — is always reflected, and the counts can never drift out of sync the
way the recorder's old in-memory "saved this session" counter did.

Data layout (see src/v2/schema.py):
    data/sequences_v2/<language>/<slug>/
        {signer}__{source}__{view}__{variant:04d}.npy   (+ .json sidecar)
    data/sequences_v2/<language>/labels.json   ->  { "<slug>": "<text>", ... }

A recorded *take* is one `variant` index for a `(signer, source)`; it is saved
as two views (clean + noisy), each an .npy + .json, i.e. up to 4 files. Deleting
a take removes all of them.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = ROOT / "data" / "sequences_v2"

# Same rule the recorder uses for language/label slugs.
ASCII_SAFE = re.compile(r"^[a-zA-Z0-9_\-]+$")

_STEM_RE = re.compile(r"^(?P<signer>.+)__(?P<source>[a-z]+)__(?P<view>[a-z]+)__(?P<variant>\d+)$")


class LibraryError(Exception):
    """A user-facing problem (bad name, missing folder, ...)."""


# ── language helpers ─────────────────────────────────────────────────

def _lang_dir(language: str) -> Path:
    return SEQUENCES / language


def _labels_path(language: str) -> Path:
    return _lang_dir(language) / "labels.json"


def _read_labels(language: str) -> dict[str, str]:
    p = _labels_path(language)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_labels(language: str, labels: dict[str, str]) -> None:
    p = _labels_path(language)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")


def list_languages() -> list[dict]:
    """Every language folder with its label + total-take counts."""
    if not SEQUENCES.exists():
        return []
    out = []
    for d in sorted(p for p in SEQUENCES.iterdir() if p.is_dir()):
        scan = scan_language(d.name)
        out.append({
            "name": d.name,
            "labels": len(scan["labels"]),
            "total_takes": sum(l["total"] for l in scan["labels"]),
        })
    return out


def create_language(name: str) -> None:
    """Make a new, empty language folder. Refuses bad names or duplicates."""
    name = (name or "").strip()
    if not ASCII_SAFE.match(name):
        raise LibraryError(
            f"language name must be ASCII letters/digits/_/- (got '{name}')")
    d = _lang_dir(name)
    if d.exists():
        raise LibraryError(f"language '{name}' already exists")
    d.mkdir(parents=True)
    _write_labels(name, {})


def delete_language(name: str) -> None:
    d = _lang_dir(name)
    if not d.exists():
        raise LibraryError(f"language '{name}' does not exist")
    shutil.rmtree(d)


# ── label helpers ────────────────────────────────────────────────────

def add_label(language: str, text: str) -> str:
    """Register a new label under a language, returning its slug. Mirrors the
    recorder's scheme: ASCII labels slug to a cleaned form, non-ASCII labels
    (e.g. Khmer) get a sequential sl_NNN slug so folder names stay portable."""
    text = (text or "").strip()
    if not text:
        raise LibraryError("label text is empty")
    if not _lang_dir(language).exists():
        raise LibraryError(f"language '{language}' does not exist")

    labels = _read_labels(language)
    for slug, existing in labels.items():
        if existing == text:
            return slug   # already there

    if ASCII_SAFE.match(text):
        base = text.lower().replace("-", "_")
        slug, i = base, 1
        while slug in labels:
            slug = f"{base}_{i}"
            i += 1
    else:
        i = 1
        while f"sl_{i:03d}" in labels:
            i += 1
        slug = f"sl_{i:03d}"

    labels[slug] = text
    _write_labels(language, labels)
    (_lang_dir(language) / slug).mkdir(parents=True, exist_ok=True)
    return slug


def delete_label(language: str, slug: str) -> None:
    """Remove a label entirely: its labels.json entry AND its folder (with any
    takes). Works even for a "ghost" label that exists only on disk (a folder
    with no labels.json entry) or only in labels.json (an entry with no folder)."""
    lang_dir = _lang_dir(language)
    if not lang_dir.exists():
        raise LibraryError(f"language '{language}' does not exist")
    labels = _read_labels(language)
    folder = lang_dir / slug
    if slug not in labels and not folder.exists():
        raise LibraryError(f"label '{slug}' does not exist in {language}")
    if slug in labels:
        del labels[slug]
        _write_labels(language, labels)
    if folder.exists():
        shutil.rmtree(folder)


# ── scanning (the single source of truth) ────────────────────────────

def _iter_takes(label_dir: Path):
    """Yield (signer, source, view, variant, path) for every parseable .npy.
    We key on .npy so a lone .json sidecar shows up as an orphan instead."""
    for p in sorted(label_dir.glob("*.npy")):
        m = _STEM_RE.match(p.stem)
        if m:
            yield (m["signer"], m["source"], m["view"],
                   int(m["variant"]), p)


def scan_language(language: str) -> dict:
    """Full picture of a language: every label with real/synthetic take counts
    and the individual real takes (for the delete UI)."""
    lang_dir = _lang_dir(language)
    labels_map = _read_labels(language)
    result_labels = []
    orphans: list[str] = []

    # union of slugs from labels.json and folders actually on disk, so a folder
    # created outside labels.json (or vice-versa) is still visible.
    slugs = set(labels_map)
    if lang_dir.exists():
        slugs |= {p.name for p in lang_dir.iterdir()
                  if p.is_dir()}

    for slug in sorted(slugs):
        label_dir = lang_dir / slug
        # take key -> set of views present, keyed (signer, source, variant)
        takes: dict[tuple, set] = {}
        if label_dir.exists():
            seen_npy_stems = set()
            for signer, source, view, variant, p in _iter_takes(label_dir):
                takes.setdefault((signer, source, variant), set()).add(view)
                seen_npy_stems.add(p.stem)
                if not p.with_suffix(".json").exists():
                    orphans.append(str(p.relative_to(SEQUENCES)))
            # a .json with no matching .npy is an orphan too
            for j in label_dir.glob("*.json"):
                if j.stem not in seen_npy_stems and _STEM_RE.match(j.stem):
                    orphans.append(str(j.relative_to(SEQUENCES)))

        real = [k for k in takes if k[1] == "real"]
        synth = [k for k in takes if k[1] == "synthetic"]
        real_takes = [
            {"signer": s, "source": src, "variant": v,
             "views": sorted(takes[(s, src, v)])}
            for (s, src, v) in sorted(real)
        ]
        result_labels.append({
            "slug": slug,
            "text": labels_map.get(slug, slug),
            "real": len(real),
            "synthetic": len(synth),
            "total": len(takes),
            "takes": real_takes,
        })

    return {"language": language, "labels": result_labels, "orphans": orphans}


# ── deletion (counts self-heal because they are scanned) ─────────────

def _delete_take_files(label_dir: Path, signer: str, source: str,
                       variant: int) -> int:
    """Remove every file of one take (both views, .npy + .json). Returns the
    number of files removed."""
    n = 0
    for view in ("clean", "noisy"):
        stem = f"{signer}__{source}__{view}__{variant:04d}"
        for ext in (".npy", ".json"):
            f = label_dir / f"{stem}{ext}"
            if f.exists():
                f.unlink()
                n += 1
    return n


def delete_take(language: str, slug: str, signer: str, variant: int,
                source: str = "real") -> int:
    label_dir = _lang_dir(language) / slug
    if not label_dir.exists():
        raise LibraryError(f"{language}/{slug} does not exist")
    removed = _delete_take_files(label_dir, signer, source, variant)
    if removed == 0:
        raise LibraryError(
            f"no take {signer} #{variant} ({source}) in {language}/{slug}")
    return removed


def delete_all(language: str, slug: str | None = None) -> int:
    """Delete every take file. If slug is given, only that label; otherwise
    every label in the language. labels.json entries are kept (the label still
    exists, just empty) — use delete_language to remove a language entirely."""
    lang_dir = _lang_dir(language)
    if not lang_dir.exists():
        raise LibraryError(f"language '{language}' does not exist")
    targets = [lang_dir / slug] if slug else [
        p for p in lang_dir.iterdir() if p.is_dir()]
    n = 0
    for d in targets:
        if not d.exists():
            continue
        for f in list(d.glob("*.npy")) + list(d.glob("*.json")):
            f.unlink()
            n += 1
    return n


def clear_synthetic(language: str, slug: str | None = None) -> int:
    """Delete only synthetic takes (all views, .npy + .json)."""
    lang_dir = _lang_dir(language)
    if not lang_dir.exists():
        raise LibraryError(f"language '{language}' does not exist")
    targets = [lang_dir / slug] if slug else [
        p for p in lang_dir.iterdir() if p.is_dir()]
    n = 0
    for d in targets:
        if not d.exists():
            continue
        for f in d.glob("*__synthetic__*"):
            f.unlink()
            n += 1
    return n
