# custom_algos — add your own algorithm

Drop a `.py` file in this folder. That is the whole process — **you never edit
any shared code**, so nobody's algorithm collides with anyone else's.

## 1. Make a file

Name it after your algorithm, e.g. `ridge.py`:

```python
from sklearn.linear_model import RidgeClassifier


def make_ridge():
    return RidgeClassifier(alpha=1.0)


ALGORITHMS = {
    "ridge": ("Ridge Classifier", make_ridge),
}
```

Three parts: import it, write a function that **returns** the model, and list it
in `ALGORITHMS` as `"key": ("Display Name", function)`.

> Pass the function itself — `make_ridge`, **not** `make_ridge()`.

## 2. Check it was picked up

```bash
python scripts/run_baseline.py --list
```

Yours should appear with your filename next to it.

## 3. Run it

```bash
python scripts/run_baseline.py --algo ridge --lang khmer_var --mode real
python scripts/run_baseline.py --algo ridge --lang khmer_var --mode both
```

Everyone runs those same two so the comparison is fair: `real` is the baseline,
`both` adds synthetic. Add `--holdout <name>` to test on a person the model
never saw.

---

## What your model has to do

Anything with scikit-learn's `.fit(X, y)` and `.predict(X)`. That covers every
sklearn estimator, a `Pipeline`, and your own class if it has those two methods.

`X` is a 2-D float array, one row per take. **It is already standardized** — the
runner wraps every algorithm in the same `StandardScaler`, so results stay
comparable and you don't need your own.

## Tuning parameters instead of adding an algorithm

Reuse a built-in key to override it with your own settings:

```python
from sklearn.ensemble import RandomForestClassifier

ALGORITHMS = {
    "rf": ("Random Forest (800 trees)", lambda: RandomForestClassifier(
        n_estimators=800, max_depth=None, random_state=0)),
}
```

Now `--algo rf` uses yours. `--list` shows the filename it came from, so it is
obvious which version ran.

## Keep results reproducible

Pass `random_state=0` to anything that takes it. Otherwise your numbers move
between runs and cannot be compared with everyone else's.

## If something goes wrong

A broken file never blocks anyone — the built-ins keep working and the error is
printed with the filename. Run `--list` to see it.

## Your own class, if you're not using sklearn

```python
import numpy as np


class NearestCentroid:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroids_ = np.stack([X[y == c].mean(axis=0)
                                    for c in self.classes_])
        return self

    def predict(self, X):
        d = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return self.classes_[d.argmin(axis=1)]


ALGORITHMS = {"centroid": ("Nearest Centroid", NearestCentroid)}
```
