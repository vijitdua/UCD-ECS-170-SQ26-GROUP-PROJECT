# Stage 2 — MLP (quick run & evaluation)

## 1. Data

Put **`train.csv`** and **`test.csv`** in `data/stage_2_data/` (format: [instructor spec](../../data/stage_2_data/README.md)).

## 2. Run

From the repo, create/activate a venv (see [requirements.txt](../../requirements.txt): `python3 -m venv .venv` then `source .venv/bin/activate` on macOS/Linux), **`cd` to this folder** (so `../../data/` and `../../result/` work), and set `PYTHONPATH` to the **repository root** (so `code.*` imports work). See the root [README.md](../../README.md) for the same pattern used in Stage 1.

**Baseline (Section 3.6 of the report)** — h=256, lr=1e-3, outputs under `result/stage_2_result/MLP_baseline/`:

```bash
PYTHONPATH=../.. python script_mlp.py
# or explicitly:
PYTHONPATH=../.. python script_mlp.py --preset baseline
```

**Ablation presets (Section 3.7)** — same data and seeds, different `Method_MLP` hyperparameters; each run writes to its own folder to avoid clobbering:

| Preset         | `script_mlp.py` flag      | What changes                         | Result folder (under `result/stage_2_result/`) |
| -------------- | --------------------------- | ------------------------------------ | --------------------------------------------- |
| A — baseline   | `--preset baseline`         | h=256, lr=1e-3 (default)             | `MLP_baseline/`                                |
| B              | `--preset hidden_128`        | `hidden_dim` 128                    | `MLP_hidden_128/`                              |
| C              | `--preset hidden_512`        | `hidden_dim` 512                    | `MLP_hidden_512/`                            |
| D              | `--preset lr_3e4`            | `learning_rate` 3e-4 (h=256)         | `MLP_lr_3e4/`                                  |

Examples:

```bash
PYTHONPATH=../.. python script_mlp.py --preset hidden_128
PYTHONPATH=../.. python script_mlp.py --preset hidden_512
PYTHONPATH=../.. python script_mlp.py --preset lr_3e4
```

## 3. Evaluation (what the script does)

- The model **trains** only on `train.csv` and is **scored** on `test.csv`
- **Metrics** are printed to the console: **accuracy** plus **precision / recall / F1** with **macro**, **weighted**, and **micro** averaging (multiclass, not binary-only).
- **Higher** accuracy / F1 (etc.) = better; read the same numbers in your report.

## 4. Saved outputs

Under the folder for your `--preset`, e.g. `result/stage_2_result/MLP_baseline/` (paths relative to repo root):

| File | What it is |
|------|------------|
| `prediction_result_0` | Pickle of run outputs (e.g. predictions) |
| `learning_curve.png` | Training loss per epoch (learning performance) |
