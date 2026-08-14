"""A working example. Copy this file, rename it, and change the three parts.

Delete this file if you don't want `ridge` showing up in --list.
"""
from sklearn.linear_model import RidgeClassifier


# 1. a function that RETURNS your model (don't fit it here)
def make_ridge():
    return RidgeClassifier(alpha=1.0, random_state=0)


# 2. list it: "key used by --algo": ("Name shown in reports", the function)
ALGORITHMS = {
    "ridge": ("Ridge Classifier", make_ridge),
}
