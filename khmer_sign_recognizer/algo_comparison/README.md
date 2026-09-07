# algo_comparison/

Experiment drivers and the reports they produce.

| | |
|---|---|
| `run_task_a.py` | **The one to use.** Computes everything for a corpus — 9 classical + 6 deep models, same-signer and unseen-signer, per-sign scores, confusions, feature importance, the split-protocol comparison, and the adversarial audit. Writes `results.json` + 6 charts. |
| `make_task_a_report.py` | Turns that `results.json` into a `.docx`. |
| `run_image_comparison.py` | Separate study: Sign Language MNIST images, not our landmarks. Exists to show algorithm rankings do **not** transfer between datasets (PROBLEM_LOG §G). |
| `make_image_report.py` | Its report. |

```bash
python algo_comparison/run_task_a.py --lang khmer_var
python algo_comparison/make_task_a_report.py --lang khmer_var
```

| flag | |
|---|---|
| `--cap-per-sign N` | use at most N takes per signer per sign — a uniform protocol for reporting without deleting anything |
| `--no-conditions` | the corpus has no lighting/distance grid (Task B is freestyle; only Task A was gridded) |
| `--tag NAME` | suffix the output folder so two runs coexist |
| `--charts-only` | redraw charts from the saved `results.json`, seconds |
| `--quick` | classical only, about a minute |

## Results folders

`results_<lang>_taskA[_<tag>]/` is the current format. Older folders
(`results/`, `results_khmer_var/`, `_both/`, `_combined/`, `results_image/`)
were produced by drivers removed in September 2026 — `run_var_experiment.py`,
`make_var_comparison.py`, `run_comparison.py`, `make_report.py`. Their `.docx`
files are kept because they record what was actually presented, but they cannot
be regenerated and their numbers **predate** the split fix, the hand-fill fix
and bone features. Do not quote them.
