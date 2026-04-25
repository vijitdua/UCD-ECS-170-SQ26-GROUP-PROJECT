# ECS 170, Spring Quarter 2026 (Prof. Jiawei Zhang), Group Project

## Group

**BogoSorters**

## Group members

- Vijit Dua (team lead)
- Pushkal Srivastava

---

## Repository layout

This repo follows the **course code template**: shared abstractions under `code/base_class/`, stage-specific implementations under `code/stage_N_code/`, runnable entry points under `script/stage_N_script/`, datasets under `data/stage_N_data/`, and outputs under `result/stage_N_result/`. Written reports (when required) can live under `reports/`. Large instructor archives may sit under `initial-archives/`.

| Path | Purpose |
|------|---------|
| `code/base_class/` | Abstract `dataset`, `method`, `setting`, `evaluate`, `result` interfaces used by every stage |
| `code/stage_1_code/` … `stage_5_code/` | Stage-specific loaders, models, settings, and evaluators |
| `script/stage_1_script/` … `stage_5_script/` | Scripts you run from the corresponding folder (e.g. Stage 1: decision tree, MLP, SVM, load results) |
| `data/stage_N_data/` | Datasets for stage *N* (large binary/csv files are often gitignored; see `.gitignore`) |
| `result/stage_N_result/` | Saved predictions, metrics, or figures for stage *N* |
| `reports/` | Stage write-ups (PDF/Markdown as your course requires) |
| `requirements.txt` | Shared Python dependencies for all stages |
| `INSTRUCTORS.md` | Full **multi-stage project overview** from the original course handout (see below) |

Empty stage folders that must stay in git use a `.gitkeep` file so Git tracks the directory.

**Stage 2 data:** `data/stage_2_data/train.csv` and `test.csv` are **not** in this repository (the training file exceeds GitHub’s file size cap). After cloning, copy them in from the **course** and read **[data/stage_2_data/README.md](data/stage_2_data/README.md)** for the full steps and for what to do if you accidentally committed the CSVs.

---

## Course project overview (instructor handout)

The official quarter-long project description—stages, grading emphasis, and setup steps—is in **[INSTRUCTORS.md](INSTRUCTORS.md)**. It was written for **ECS 189G (Winter 2024)**; align deadlines and deliverables with **ECS 170 / Canvas** for Spring 2026.

---

## Python environment (one venv for all stages)

Use a **single** virtual environment at the repo root so PyTorch, scikit-learn, and later add-ons (e.g. PyTorch Geometric for Stage 5) stay consistent.

```bash
cd /path/to/UCD-ECS-170-SQ26-GROUP-PROJECT
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

**GPU / CUDA:** if you need a specific PyTorch+CUDA build, install `torch` / `torchvision` from the selector at [pytorch.org](https://pytorch.org/get-started/locally/) instead of the defaults pulled by `requirements.txt`.

---

## Running scripts from the terminal (Stage 1 and the same template pattern)

The course template does two things that matter when you use **`python`** from a shell (PyCharm often hides this by configuring the project for you):

1. **Imports** — Scripts use packages such as `code.stage_1_code`. Python must see the **repository root** (the directory that contains the `code/` folder) on `PYTHONPATH`. If it does not, you may get `ModuleNotFoundError` or `'code' is not a package` because the standard library also exposes a top-level module named `code`.

2. **Data and result paths** — Scripts pass relative paths like `../../data/stage_1_data/...` and `../../result/stage_1_result/...`. Those are resolved from the process **current working directory** (where you ran the command), not from the script file’s folder. So your shell **cwd must be** `script/stage_1_script/` (two levels below the repo root), or those paths will point outside the project and you will see `FileNotFoundError`.

In particular, running `PYTHONPATH=. python script/stage_1_script/script_svm.py` **from the repo root** fixes imports but **breaks** the `../../...` file paths, because from the repo root `../..` goes above the repository.

**Recommended commands** (repo root → venv → stage folder, then run with `PYTHONPATH` pointing at the repo root):

```bash
cd /path/to/UCD-ECS-170-SQ26-GROUP-PROJECT
source .venv/bin/activate
cd script/stage_1_script
PYTHONPATH=../.. python script_decision_tree.py   # example
PYTHONPATH=../.. python script_mlp.py
PYTHONPATH=../.. python script_svm.py
PYTHONPATH=../.. python script_load_result.py
```

You can instead `export PYTHONPATH=/path/to/UCD-ECS-170-SQ26-GROUP-PROJECT` once per shell, then use plain `python script_svm.py` after `cd script/stage_1_script`.

Outputs are written under `result/stage_1_result/` when the template runs successfully. Later stages that reuse the same `from code.stage_N_code...` and `../../data/...` layout need the same **repo root on `PYTHONPATH`** and **cwd in the matching `script/stage_N_script/`** unless you change those paths in code.

**Stage 2 (MLP on pre-partitioned MNIST-style CSVs):** `cd script/stage_2_script` with the same `PYTHONPATH=../..` pattern, then e.g. `python script_mlp.py` (baseline) or `python script_mlp.py --preset hidden_128` (ablation). **Details, preset table, and output locations** for the report are in [script/stage_2_script/README.md](script/stage_2_script/README.md).
