# ECS 170, Spring Quarter 2026 (Prof. Jiawei Zhang), Group Project

## Group:

BogoSorters

## Group Members:

- Vijit Dua (Team Lead)
- Pushkal Srivastava

## Python environment (one venv for all stages)

Use a **single** virtual environment at the repo root so PyTorch, scikit-learn, and later add-ons (e.g. PyTorch Geometric for Stage 5) stay consistent.

```bash
cd /path/to/UCD-ECS-170-SQ26-GROUP-PROJECT
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python scripts/verify_env.py
```

**GPU / CUDA:** if you need a specific PyTorch+CUDA build, install `torch` / `torchvision` from the selector at [pytorch.org](https://pytorch.org/get-started/locally/) instead of the defaults pulled by `requirements.txt`.
