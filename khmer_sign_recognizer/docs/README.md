# Documentation index

Twenty-two documents, in five folders. Find your row in the first table and go
there directly.

| I want to… | Read |
|---|---|
| install this on my machine | [setup/](#setup) for your OS |
| record my takes for the team | [guides/TEAM_DATA_COLLECTION_PLAN.md](guides/TEAM_DATA_COLLECTION_PLAN.md) |
| look up a command | [guides/COMMANDS.md](guides/COMMANDS.md) |
| understand how the system works | [reference/SYSTEM_ARCHITECTURE.md](reference/SYSTEM_ARCHITECTURE.md) |
| explain the results to a teacher | [results/TASK_A_FOR_TEACHER.md](results/TASK_A_FOR_TEACHER.md) |
| know what synthetic data actually does | [reference/SYNTHETIC_RETARGETING.md](reference/SYNTHETIC_RETARGETING.md) |
| find out why something is built the way it is | [project/PROBLEM_LOG.md](project/PROBLEM_LOG.md) |
| **write the paper** | [project/PAPER_DRAFT.md](project/PAPER_DRAFT.md) |
| **know the latest measured results** | [project/PROBLEM_LOG.md](project/PROBLEM_LOG.md) §J–N, and `Task_A_Report.docx` |

---

## setup/

Getting a working install. One file per platform — pick yours and ignore the
rest.

| | |
|---|---|
| [LINUX_REBUILD.md](setup/LINUX_REBUILD.md) | Linux (CachyOS / Arch). The reference install: Python 3.11, GPU onnxruntime, Wayland notes. Most current of the three. |
| [SETUP_WINDOWS.md](setup/SETUP_WINDOWS.md) | Windows. Same code, different install steps. |
| [SETUP_MACOS.md](setup/SETUP_MACOS.md) | macOS, Intel and Apple Silicon. Everything runs on CPU — fine for recording, sluggish for live Recognize. |
| [FIRST_TIME_COLAB.md](setup/FIRST_TIME_COLAB.md) | Training on Colab, end to end. ⚠️ Written for the Drive + deep-model workflow; predates the local classical-ML pipeline. |

## guides/

Doing the work.

| | |
|---|---|
| [QUICKSTART.md](guides/QUICKSTART.md) | Zero to your first recording. Read this before the others. |
| [COMMANDS.md](guides/COMMANDS.md) | Every command the project uses, grouped by task. The one to keep open. |
| [TEAM_DATA_COLLECTION_PLAN.md](guides/TEAM_DATA_COLLECTION_PLAN.md) | The recording assignment: Task A's 12-take variation grid and Task B's freestyle pool, plus how folders get merged. |
| [TEAM_ONBOARDING.md](guides/TEAM_ONBOARDING.md) | What a new teammate does on day one. |
| [TEAM_SETUP.md](guides/TEAM_SETUP.md) | Recording conventions and the label discipline that keeps merges clean. |

## reference/

How it works and why.

| | |
|---|---|
| [SYSTEM_ARCHITECTURE.md](reference/SYSTEM_ARCHITECTURE.md) | The full architecture: capture → normalise → store → train → infer, with the data contract and the reasoning behind each choice. |
| [SYNTHETIC_RETARGETING.md](reference/SYNTHETIC_RETARGETING.md) | How one take becomes several takes on differently-proportioned bodies, and the proof the sign survives it. |
| [WORKFLOW.md](reference/WORKFLOW.md) | The 8-task team split, data pipeline and hard rules. ⚠️ The task list is from term 1 — several items are now done or dropped. |

## results/

What we measured.

| | |
|---|---|
| **Task_A_Report.docx** | `algo_comparison/results_khmer_var_taskA/` — the current report: all five questions, 15 models, the audit. Regenerate with `run_task_a.py` then `make_task_a_report.py`. |
| [TASK_A_FOR_TEACHER.md](results/TASK_A_FOR_TEACHER.md) | The variation experiment in the order you would say it out loud. Start here. |
| [TASK_A_EXPLAINED.md](results/TASK_A_EXPLAINED.md) | `Task_A_Report.docx` walked through section by section. |
| [SYNTHETIC_EXPLAINED.md](results/SYNTHETIC_EXPLAINED.md) | The real vs real+synthetic comparison, and why the gain is only about 2 points. |
| [EXPERIMENT_REPORT.md](results/EXPERIMENT_REPORT.md) | Algorithm comparison and robustness on the 30-take corpus, including the same-signer → unseen-signer collapse. |
| [FINDINGS.md](results/FINDINGS.md) | Engineering findings: what broke, what fixed it, what conditions the system works under. |
| [EXPERIMENT_PROTOCOL.md](results/EXPERIMENT_PROTOCOL.md) | The original synthetic-data protocol. ⚠️ Historical — written to run on AUTSL before there were Khmer recordings. |

## project/

Planning and history.

| | |
|---|---|
| [PAPER_DRAFT.md](project/PAPER_DRAFT.md) | Research-paper content draft — the argument, every table with the file it came from, and the gaps that still block writing. |
| [TIMELINE.md](project/TIMELINE.md) | The two-year plan; every milestone has a hard deliverable. |
| [PROBLEM_LOG.md](project/PROBLEM_LOG.md) | Every problem hit and every change made, with cause, fix and evidence. The most useful file when picking work back up. **§J–N are the current results.** |
| [PROJECT_BRIEF_FOR_AI.md](project/PROJECT_BRIEF_FOR_AI.md) | Self-contained project description to paste into an AI assistant for literature work. |
| [PRESENTATION_TERM1.md](project/PRESENTATION_TERM1.md) | Slide content for the term-1 review. ⚠️ Historical — the status claims are from before the web app and the experiments landed. |

---

## A note on the ⚠️ marks

Four documents describe a stage the project has moved past. They are kept
because they record real decisions and the reasoning behind them, but do not
follow their instructions — check the current path in `COMMANDS.md` first.

The largest retired piece is the original **Windows → WSL → Godot** pipeline,
which streamed landmarks over UDP to a Godot 4.6 mannequin. It is gone; the
3D view is now an in-process Open3D window, and there is no UDP, no WSL layer
and no Godot anywhere in the code. Where an old document mentions any of
those, that section no longer applies.
