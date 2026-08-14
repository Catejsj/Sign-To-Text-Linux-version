# Sign Language MNIST — Algorithm Lab

Train **one algorithm at a time** on the Sign Language MNIST dataset, then
combine everyone's results into a single comparison chart.

Works on **Windows, macOS and Linux**, and in **Google Colab** (no install at all).

---

## Two ways to run it

| | best for | install needed |
|---|---|---|
| **A — Google Colab** | Mac users, anyone without Python, guaranteeing everyone gets the same numbers | none |
| **B — On your computer** | Windows/Linux users who already have Python | Python 3.9+ |

Both produce the same `results/<algo>.json` files, so the team can mix and match.

---

## A. Google Colab (recommended)

1. Go to [colab.research.google.com](https://colab.research.google.com) →
   **File → Upload notebook** → choose `SignLang_Image_Lab_Colab.ipynb`.
2. Run **Step 1** and **Step 2** (Step 2 asks you to upload `archive.zip`).
3. In **Step 4**, set `MY_ALGORITHM` to the one you were assigned.
4. Run **Step 5** to train, **Step 6** to download your result.

Everything runs on Google's servers, so a Mac, a Windows laptop and a Chromebook
all produce **identical numbers** — the results differ only by algorithm, not by
whose machine ran it.

For the neural models (`gru`, `cnn`), switch on the GPU first:
*Runtime → Change runtime type → T4 GPU*.

**The GPU only helps `gru` and `cnn`.** Everything else is scikit-learn, which is
CPU-only, so the runtime type makes no difference to it.

Colab's free tier has about 2 CPU cores, so classical algorithms run far slower
there than on a desktop: `gboost` and `svm` take 4-8 minutes, `logreg` 5-10, and
`mlp` 10-20. That is normal — not a hang. Do not lower `TRAIN_SIZE` to speed it
up, or your result stops being comparable with everyone else's.

---

## B. On your own computer

### 1. Install

**Windows** (Command Prompt):
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Only for the `gru` / `cnn` algorithms: `pip install torch`

### 2. Get the dataset

Download it from
[Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
(free Kaggle account) and drop **`archive.zip` into the `data/` folder**.
No need to unzip — it is found automatically.

### 3. Run

```bash
python run.py --list          # show available algorithms
python run.py --algo lda      # train one algorithm
python run.py --chart         # build charts from all results
```

---

## Commands

| command | what it does |
|---|---|
| `python run.py --list` | list algorithms you can run |
| `python run.py --algo svm` | train one algorithm, save `results/svm.json` |
| `python run.py --all` | train every algorithm (slow) |
| `python run.py --chart` | combine all `results/*.json` into charts + `summary.json` |
| `python run.py --algo mlp --train-size 5000` | train on a subset (faster; say so in your report) |
| `python run.py --algo rf --data path/to/archive.zip` | use a dataset somewhere else |

Available keys: `lda svm logreg knn gboost mlp rf nb tree gru cnn`

---

## Adding your own algorithm

Edit **`algorithms.py`** (or the Step 3 cell in the notebook). Two steps:

```python
# 1. write a factory function
def make_ridge():
    from sklearn.linear_model import RidgeClassifier
    return RidgeClassifier()

# 2. register it
ALGORITHMS = {
    ...
    "ridge": ("Ridge Classifier", make_ridge),
}
```

Then `python run.py --algo ridge`.

Anything with scikit-learn's `.fit(X, y)` / `.predict(X)` interface works —
including your own class, or a PyTorch model wrapped like `torch_models.py` does.

- `X` is `(n_images, 784)` float32, pixel values scaled to **0–1**
- `y` is `(n_images,)` int — the letter class

Nothing else in the project needs changing.

---

## Working as a team

1. Each person runs **one** algorithm and gets `results/<algo>.json`.
2. Everyone sends their `.json` to one person.
3. That person drops all the files into `results/` and runs
   `python run.py --chart` (or Step 7 in the notebook).

Result: one chart and one table covering every algorithm, all evaluated on the
same 7,172 test images.

---

## What gets produced

```
results/
  <algo>.json           accuracy, macro-F1, timings, confusion matrix
  samples.png           example images from the dataset
  confusion_best.png    confusion matrix of the best algorithm
  comparison_chart.png  the grouped bar chart
  summary.json          every run, ranked
```

---

## About the dataset

- 27,455 training and 7,172 test images, 28×28 grayscale.
- 24 classes — the letters **J and Z are excluded** because they are signed with
  motion, and a still image cannot represent them. That is a genuine limitation
  of image-based sign recognition and is worth mentioning in a report.
- The **official train/test split is used unchanged**, so every algorithm is
  scored on exactly the same images.
- Pixels are scaled to 0–1. No other preprocessing is applied.

**Metrics.** Accuracy is the percentage of test images classified correctly.
Macro-F1 averages the F1 score of every letter equally, so a rare letter counts
as much as a common one — if accuracy is much higher than macro-F1, the model is
doing well on frequent classes and poorly on rare ones.

**One thing worth noting in your write-up:** every classical algorithm here
treats the 784 pixels as *independent features* and ignores the 2D layout of the
image. The included `cnn` does use spatial structure, which is why it scores much
higher — a useful contrast to discuss.

---

## Troubleshooting

| problem | fix |
|---|---|
| `No dataset found` | put `archive.zip` in the `data/` folder |
| `No module named sklearn` | activate the venv, then `pip install -r requirements.txt` |
| `gru`/`cnn` fail | `pip install torch` (or use Colab with a GPU runtime) |
| SVM or k-NN too slow | add `--train-size 5000` |
| `python` not found (Windows) | try `py` instead of `python` |
