"""The algorithms `scripts/run_baseline.py` can train, and how to add your own.

ADDING YOUR OWN — you do NOT need to edit this file
---------------------------------------------------
Make a file in `custom_algos/` named anything you like, e.g.
`custom_algos/ridge.py`:

    from sklearn.linear_model import RidgeClassifier

    def make_ridge():
        return RidgeClassifier(alpha=1.0)

    ALGORITHMS = {
        "ridge": ("Ridge Classifier", make_ridge),
    }

That is the whole thing. It is picked up automatically:

    python scripts/run_baseline.py --list
    python scripts/run_baseline.py --algo ridge --lang khmer_var

Because everyone works in their own file, nobody's addition collides with
anyone else's when the folder is shared.

WHAT A FACTORY MUST RETURN
--------------------------
Anything with scikit-learn's `.fit(X, y)` / `.predict(X)` — every sklearn
estimator, a `Pipeline`, or your own class with those two methods.

X arrives as a 2-D float array, one row per take, and is already standardized
(see `wrap` below), so you do not need your own scaler.
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUSTOM_DIR = ROOT / "custom_algos"


# ── built-ins ────────────────────────────────────────────────────────

def make_knn():
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=5)


def make_logreg():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000)


def make_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, random_state=0)


def make_svm():
    from sklearn.svm import SVC
    return SVC(kernel="rbf", C=10, gamma="scale")


def make_nb():
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB()


def make_tree():
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(random_state=0)


def make_gboost():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(random_state=0)


def make_lda():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis()


def make_mlp():
    # A small neural net. Light deep-learning — check with the teacher if
    # "simple algorithms only" is meant to exclude it.
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=800,
                         random_state=0)


BUILTIN: dict = {
    "knn":    ("k-NN", make_knn),
    "logreg": ("Logistic Regression", make_logreg),
    "rf":     ("Random Forest", make_rf),
    "svm":    ("SVM", make_svm),
    "nb":     ("Naive Bayes", make_nb),
    "tree":   ("Decision Tree", make_tree),
    "gboost": ("Gradient Boosting", make_gboost),
    "lda":    ("LDA", make_lda),
    "mlp":    ("MLP", make_mlp),
}


# ── custom_algos/ ────────────────────────────────────────────────────

def _load_custom() -> tuple[dict, list[str]]:
    """Import every custom_algos/*.py and collect their ALGORITHMS dicts.

    A broken file must not stop the built-ins from working, so failures are
    collected and reported rather than raised.
    """
    found: dict = {}
    errors: list[str] = []
    if not CUSTOM_DIR.is_dir():
        return found, errors

    for py in sorted(CUSTOM_DIR.glob("*.py")):
        if py.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                f"custom_algos.{py.stem}", py)
            if spec is None or spec.loader is None:
                raise ImportError("could not load")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
        except Exception:                                     # noqa: BLE001
            errors.append(f"{py.name}:\n"
                          + "".join(traceback.format_exc(limit=2)).rstrip())
            continue

        table = getattr(mod, "ALGORITHMS", None)
        if not isinstance(table, dict):
            errors.append(f"{py.name}: no ALGORITHMS dict — add one, e.g.\n"
                          f'    ALGORITHMS = {{"myalgo": ("My Algo", make_myalgo)}}')
            continue
        for key, value in table.items():
            try:
                label, factory = value
            except (TypeError, ValueError):
                errors.append(f"{py.name}: ALGORITHMS[{key!r}] should be "
                              f'("Display Name", factory_function)')
                continue
            if not callable(factory):
                errors.append(f"{py.name}: ALGORITHMS[{key!r}] factory is not "
                              f"a function — pass the function itself, not a "
                              f"call to it (make_x, not make_x())")
                continue
            found[key] = (label, factory, py.name)
    return found, errors


def registry() -> tuple[dict, list[str]]:
    """All algorithms: {key: (label, factory, origin)} plus any load errors.

    A custom algorithm may deliberately override a built-in of the same name —
    handy for trying different parameters without renaming anything.
    """
    table = {k: (label, fn, "built-in") for k, (label, fn) in BUILTIN.items()}
    custom, errors = _load_custom()
    table.update(custom)
    return table, errors


def wrap(estimator):
    """Standardize features, then fit the estimator. Same for everyone, so
    results stay comparable."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), estimator)


def build_model(algo: str):
    table, errors = registry()
    if algo not in table:
        lines = [f"unknown --algo {algo!r}", "", "available:"]
        for key, (label, _fn, origin) in sorted(table.items()):
            lines.append(f"  {key:<10} {label}  ({origin})")
        lines += ["", "Add your own: put a file in custom_algos/ — see "
                      "src/v2/algorithms.py for a 6-line example."]
        if errors:
            lines += ["", "NOTE: some custom_algos files failed to load:"]
            lines += [f"  {e}" for e in errors]
        raise SystemExit("\n".join(lines))

    label, factory, origin = table[algo]
    for e in errors:
        print(f"[custom_algos] skipped — {e}\n")
    return wrap(factory()), label, origin


def print_list() -> None:
    table, errors = registry()
    print("algorithms available to --algo\n")
    width = max(len(k) for k in table)
    for key, (label, _fn, origin) in sorted(table.items(),
                                            key=lambda kv: (kv[1][2] != "built-in",
                                                            kv[0])):
        print(f"  {key:<{width}}  {label:<22} {origin}")
    print(f"\ncustom_algos/ folder: {CUSTOM_DIR}")
    if errors:
        print("\nfiles that failed to load:")
        for e in errors:
            print(f"  {e}")
    else:
        print("Drop a .py file in there to add your own — no need to edit "
              "any shared code.")
