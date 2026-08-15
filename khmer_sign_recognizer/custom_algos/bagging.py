"""Bagged decision trees — measured best of the simple algorithms here.

Not one of the project's built-in nine. Added because it was the strongest
candidate tested on our own data:

  khmer, leave-one-signer-out (5 seeds, macro-F1)
      bagging  75.5  [71.4, 79.3]   <- bootstrap 95% interval
      ridge    61.9  [57.3, 66.2]
      lda      60.9  [56.0, 65.5]
      rf       57.6  [52.5, 62.1]

The intervals for bagging and random forest do not overlap, so the ~18-point
lead on unseen signers is not noise.

Why it beats random forest here: RF also samples a random subset of FEATURES at
every split. With 576 summary features from only 48 joints, many carry the same
information, and that extra randomness costs more than it adds. Bagging keeps
the bootstrap sampling of takes but lets each tree use every feature.
"""
from sklearn.ensemble import BaggingClassifier


def make_bagging():
    return BaggingClassifier(n_estimators=100, random_state=0)


ALGORITHMS = {
    "bagging": ("Bagged Trees", make_bagging),
}
