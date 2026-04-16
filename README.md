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

## Running Stage 1

From the repo root, activate the venv, then:

```bash
cd script/stage_1_script
python script_decision_tree.py   # example
python script_mlp.py
python script_svm.py
```

Outputs are written under `result/stage_1_result/` when the template runs successfully.
