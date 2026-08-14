"""Algorithm registry — ADD YOUR ALGORITHM HERE.

Every algorithm is one entry in ALGORITHMS. To add your own, write a function
that returns a model with scikit-learn's `.fit(X, y)` / `.predict(X)` interface
and register it at the bottom.

    X is (n_images, 784) float32, pixel values scaled to 0-1
    y is (n_images,)     int, the letter class

Nothing else in the project needs to change.
"""
from __future__ import annotations

# One seed for every algorithm that has any randomness. Without this,
# HistGradientBoosting alone varied by ~1.4 points between identical runs,
# which is enough to change the ranking.
SEED = 0


# ── the algorithms already included ──────────────────────────────────

def make_lda():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis()


def make_svm():
    from sklearn.svm import SVC
    return SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED)


def make_logreg():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, random_state=SEED)


def make_knn():
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)


def make_gboost():
    # HistGradientBoosting, not GradientBoosting: the latter trains
    # (n_classes x n_estimators) trees and takes hours on 784 features.
    #
    # early_stopping is pinned to False on purpose. The default ('auto') turns
    # itself ON above 10,000 samples and then holds back 10% of the training
    # data as an internal validation set -- so the model trains on less data and
    # scores ~0.6 points lower. Worse, whether that happens depends on the
    # scikit-learn version, so Colab and a local machine disagreed. Pinning it
    # makes the result the same everywhere.
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=150, random_state=SEED,
                                          early_stopping=False)


def make_mlp():
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=60,
                         random_state=SEED)


def make_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)


def make_nb():
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB()


def make_tree():
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(random_state=SEED)


def make_gru():
    """Bidirectional GRU: reads each image as 28 rows of 28 pixels.

    Wrapped so it exposes the same fit/predict interface as the scikit-learn
    models. Requires PyTorch; if torch is missing this algorithm is skipped.
    """
    from torch_models import GRUClassifier
    return GRUClassifier(hidden=128, epochs=25, lr=2e-3, batch_size=256)


def make_cnn():
    """Small convolutional network — the only model here that uses the 2D
    structure of the image rather than treating pixels as independent."""
    from torch_models import CNNClassifier
    return CNNClassifier(epochs=12, lr=1e-3, batch_size=128)


# ── registry ─────────────────────────────────────────────────────────
#
#  key      -> (display name, factory function)
#
#  ADD YOUR ALGORITHM: write a make_*() above, then add one line here.
#  Example:
#      "ridge": ("Ridge Classifier", make_ridge),
#
ALGORITHMS = {
    "lda":    ("LDA", make_lda),
    "svm":    ("SVM", make_svm),
    "logreg": ("Logistic Regression", make_logreg),
    "knn":    ("k-NN", make_knn),
    "gboost": ("Gradient Boosting", make_gboost),
    "mlp":    ("MLP", make_mlp),
    "rf":     ("Random Forest", make_rf),
    "nb":     ("Naive Bayes", make_nb),
    "tree":   ("Decision Tree", make_tree),
    "gru":    ("GRU (recurrent)", make_gru),
    "cnn":    ("CNN (convolutional)", make_cnn),
}


def get(name: str):
    """Return (display_name, model_instance) for a registry key."""
    if name not in ALGORITHMS:
        raise SystemExit(
            f"Unknown algorithm: {name!r}\n"
            f"Available: {', '.join(sorted(ALGORITHMS))}\n"
            f"Add your own in algorithms.py")
    label, factory = ALGORITHMS[name]
    return label, factory()


def display_name(name: str) -> str:
    return ALGORITHMS.get(name, (name, None))[0]
