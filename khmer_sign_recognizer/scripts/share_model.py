"""Package a trained model bundle for someone else to run.

A bundle is one `.joblib` in `models/recognizers/`, and it already carries
everything needed to rebuild its own input — labels, Khmer text, feature mode
and the sequence spec. So sharing is mostly a copy.

What this adds is the checking. A bundle is a *pickle*: it stores live Python
objects, so it only loads where the same classes and roughly the same library
versions exist. This script loads the bundle the way the recipient's machine
will, records the versions it was built against, and writes a README so they
know what they are getting and what it needs.

    python scripts/share_model.py khmer tcn
    python scripts/share_model.py --list

Writes to `share/<language>__<algo>/`.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.v2.recognizer import bundle_path, load_bundle, list_models   # noqa: E402


def versions() -> dict:
    import numpy, joblib
    out = {"python": sys.version.split()[0], "numpy": numpy.__version__,
           "joblib": joblib.__version__}
    try:
        import sklearn; out["scikit-learn"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import torch; out["torch"] = torch.__version__
    except ImportError:
        pass
    return out


README = """# {lang} · {algo} — SignLink model

Drop `{fname}` into:

    khmer_sign_recognizer/models/recognizers/

Then start the control panel, switch to **Recognize**, and pick
**{lang} · {algo}** from the model list. Nothing else to configure — the
bundle carries its own labels and feature settings.

## What is in it

| | |
|---|---|
| Signs | {nclass} |
| Feature mode | `{fmode}` |
| Sequence spec | `{spec}` |
| Trained | {saved} |
| Model class | `{mtype}` |
| Size | {size:.1f} MB |

Signs it knows: {labels}

## What your machine needs

This file stores live Python objects, so it needs a similar environment to the
one that built it:

{vertable}

The same **repository code** is also required — the model class
(`{mtype}`) is imported by name when the file is opened. A bundle
cannot be loaded without the project it came from.

Close versions are usually fine. Very different ones (numpy 1.x against 2.x,
or a much older scikit-learn) will fail or warn.

## ⚠️ About the accuracy in the metadata

The bundle records `{acckey} = {accval}`. **Do not quote that as the
project's accuracy.** It is measured on data this model was trained on.

The honest figure, measured on a signer the model had never seen, is
**92.0%**. The high number in the metadata exists because shipped models are
deliberately trained on every recording for the best real-world behaviour —
which makes their own score meaningless as a benchmark.

## ⚠️ Only open bundles you trust

Loading a `.joblib` runs code inside it. Treat one like a program, not a
document — open them from people you know, and do not accept them from
strangers.

## If it does not work

Run the checker from the repository root **before** troubleshooting anything
else. It says exactly what is wrong:

    python check_model.py
"""


# Sent alongside the bundle. The receiver runs it from the repo root; it
# reports the precise reason a load fails instead of leaving them with a
# stack trace from deep inside joblib.
CHECKER = '''"""Check that this machine can load the shared SignLink model.

Run from the repository root:

    python check_model.py

Or with the project venv:

    ./venv/bin/python check_model.py
"""
import sys
from pathlib import Path

BUILT_WITH = {built!r}
BUNDLE = "{fname}"

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in [HERE, *HERE.parents]
             if (p / "src" / "v2" / "recognizer.py").exists()), None)

print("SignLink model check\\n" + "-" * 40)

if ROOT is None:
    print("FAIL  Not inside the SignLink repository.")
    print("      Put this file and the .joblib inside the project folder,")
    print("      or clone the repo first — the model cannot load without it.")
    sys.exit(1)
print(f"repo        : {{ROOT}}")
sys.path.insert(0, str(ROOT))

bundle = HERE / BUNDLE
if not bundle.exists():
    bundle = ROOT / "models" / "recognizers" / BUNDLE
if not bundle.exists():
    print(f"FAIL  {{BUNDLE}} not found next to this script or in models/recognizers/")
    sys.exit(1)
print(f"bundle      : {{bundle}}  ({{bundle.stat().st_size / 1e6:.1f}} MB)")

# version comparison first — a mismatch here explains most failures
print("\\nversions:")
# the pip name and the import name differ for some packages
IMPORT_NAME = {{"scikit-learn": "sklearn"}}
missing, differs = [], []
for pkg, built in BUILT_WITH.items():
    if pkg == "python":
        have = sys.version.split()[0]
    else:
        try:
            mod = __import__(IMPORT_NAME.get(pkg, pkg.replace("-", "_")))
            have = getattr(mod, "__version__", "?")
        except ImportError:
            have = None
    if have is None:
        print(f"  {{pkg:<14}} NOT INSTALLED  (built with {{built}})")
        missing.append(pkg)
    else:
        major_differs = have.split(".")[0] != built.split(".")[0]
        flag = "  <-- major difference" if major_differs else ""
        print(f"  {{pkg:<14}} {{have:<14}} (built with {{built}}){{flag}}")
        if major_differs:
            differs.append(pkg)

print()
try:
    import joblib
    b = joblib.load(bundle)
except ModuleNotFoundError as e:
    print(f"FAIL  Missing Python module: {{e.name}}")
    print("      Install the project requirements into your environment.")
    sys.exit(1)
except AttributeError as e:
    print(f"FAIL  The model class could not be rebuilt: {{e}}")
    print("      Your copy of the repo is a different version from the sender's.")
    print("      Pull the same commit and try again.")
    sys.exit(1)
except Exception as e:
    print(f"FAIL  {{type(e).__name__}}: {{e}}")
    if missing:
        print(f"      Likely cause: {{', '.join(missing)}} not installed.")
    elif differs:
        print(f"      Likely cause: version mismatch in {{', '.join(differs)}}.")
    sys.exit(1)

m = b["model"]
print("OK    bundle loaded")
print(f"      model    : {{type(m).__module__}}.{{type(m).__name__}}")
print(f"      signs    : {{len(b.get('label_to_idx', {{}}))}}")
print(f"      features : {{b.get('feature_mode')}}")
print(f"      text     : {{' '.join(b.get('labels_text', {{}}).values())}}")

dest = ROOT / "models" / "recognizers"
if bundle.parent != dest:
    print(f"\\nNext: copy {{BUNDLE}} into {{dest}}")
    print("      then start the app, switch to Recognize, and pick it from the list.")
else:
    print("\\nAlready in models/recognizers/ — start the app and pick it in Recognize.")

if differs:
    print(f"\\nNote: {{', '.join(differs)}} differ by a major version from the build")
    print("      machine. It loaded anyway, but sanity-check a few predictions.")
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("language", nargs="?", help="e.g. khmer")
    ap.add_argument("algo", nargs="?", help="e.g. tcn, gru, rf")
    ap.add_argument("--list", action="store_true", help="show saved models")
    args = ap.parse_args()

    if args.list or not (args.language and args.algo):
        print("Saved models:\n")
        for m in list_models():
            print(f"  {m['language']:<10} {m['algo']:<12} {m.get('name', '')}")
        print("\nUsage: python scripts/share_model.py <language> <algo>")
        return

    src = bundle_path(args.language, args.algo)
    if not src.exists():
        raise SystemExit(f"no such bundle: {src}")

    # Load it the way the recipient will. If this raises here, it will raise
    # for them too — better to find out now than after sending it.
    b = load_bundle(src)
    model = b["model"]
    mtype = f"{type(model).__module__}.{type(model).__name__}"
    meta = b.get("meta", {})

    acckey = next((k for k in ("accuracy", "eval_accuracy", "eval_macro_f1")
                   if k in meta), None)
    accval = f"{meta[acckey]:.4f}" if acckey else "n/a"

    out = ROOT / "share" / f"{args.language}__{args.algo}"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out / src.name)

    v = versions()
    vertable = "\n".join(f"| {k} | {val} |" for k, val in v.items())
    vertable = "| package | version used to build |\n|---|---|\n" + vertable

    (out / "README.md").write_text(README.format(
        lang=args.language, algo=args.algo, fname=src.name,
        nclass=len(b.get("label_to_idx", {})),
        fmode=b.get("feature_mode"), spec=b.get("sequence_spec") or "n/a",
        saved=meta.get("saved_at", "unknown"), mtype=mtype,
        size=src.stat().st_size / 1e6,
        labels=" ".join(b.get("labels_text", {}).values()) or "n/a",
        vertable=vertable, acckey=acckey or "accuracy", accval=accval,
    ))

    (out / "check_model.py").write_text(CHECKER.format(built=v, fname=src.name))

    # One archive is easier to send over chat than three loose files, and it
    # survives apps that rewrite or strip unknown attachments.
    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent,
                                  base_dir=out.name)

    print(f"loaded OK   : {src.name}  ({src.stat().st_size / 1e6:.1f} MB)")
    print(f"model class : {mtype}")
    print(f"feature mode: {b.get('feature_mode')}  spec={b.get('sequence_spec') or '{}'}")
    print(f"signs       : {len(b.get('label_to_idx', {}))}")
    print()
    print(f"folder      : {out}")
    print(f"send this   : {archive}  ({Path(archive).stat().st_size / 1e6:.1f} MB)")
    print("              contains the model, a README, and check_model.py")


if __name__ == "__main__":
    main()
