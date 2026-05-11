# Stage 3 — CNN image classification (quick run & evaluation)

## 1. Data

Download the Stage 3 dataset archive from the course page and place the three binary pickle files under `data/stage_3_data/`:

```
data/stage_3_data/
    MNIST       ← handwritten digit images (28×28 grayscale)
    ORL         ← human face images (112×92, grayscale stored as 3-ch)
    CIFAR       ← colored object images (32×32 RGB)
```

The files are already ignored by `.gitignore` (large binaries). `script_data_loader.py` in the same folder lets you inspect individual images.

## 2. Run

### One command (all presets, sequential)

From the **repository root** (the folder that contains `code/`, `data/`, `script/`). This runs **every** Stage 3 preset one after another (MNIST → ORL → CIFAR; baseline, then `lr_3e4`, then deeper for each). Uses `-u` so log lines show up immediately.

```bash
PYTHONPATH=. .venv/bin/python -u script/stage_3_script/script_cnn.py --preset all
```

If your venv path differs or you prefer activating first:

```bash
source .venv/bin/activate && PYTHONPATH=. python -u script/stage_3_script/script_cnn.py --preset all
```

If one preset crashes (e.g. missing data file), the script **prints the traceback**, **continues** with the next preset, then exits with code **1** and lists failed names.

For **single** presets, still run from repo root with `PYTHONPATH=.` and `**python -u`** (or `.venv/bin/python -u`) so logs stream immediately.

**Baseline runs (one per dataset — Sections 3.5 & 3.6 of the report):**

```bash
PYTHONPATH=. python -u script/stage_3_script/script_cnn.py --preset MNIST_baseline
PYTHONPATH=. python -u script/stage_3_script/script_cnn.py --preset ORL_baseline
PYTHONPATH=. python -u script/stage_3_script/script_cnn.py --preset CIFAR_baseline
```

**Ablation presets (Section 3.7) — same data and seeds, different hyperparameters:**


| Preset             | `--preset` flag  | What changes           | Result folder                               |
| ------------------ | ---------------- | ---------------------- | ------------------------------------------- |
| MNIST A — baseline | `MNIST_baseline` | 2-block CNN, lr=1e-3   | `result/stage_3_result/MNIST/CNN_baseline/` |
| MNIST B            | `MNIST_lr_3e4`   | lr=3e-4                | `result/stage_3_result/MNIST/CNN_lr_3e4/`   |
| MNIST C            | `MNIST_deeper`   | 3 conv blocks, lr=1e-3 | `result/stage_3_result/MNIST/CNN_deeper/`   |
| ORL A — baseline   | `ORL_baseline`   | 2-block CNN, lr=1e-3   | `result/stage_3_result/ORL/CNN_baseline/`   |
| ORL B              | `ORL_lr_3e4`     | lr=3e-4                | `result/stage_3_result/ORL/CNN_lr_3e4/`     |
| ORL C              | `ORL_deeper`     | 3 conv blocks, lr=1e-3 | `result/stage_3_result/ORL/CNN_deeper/`     |
| CIFAR A — baseline | `CIFAR_baseline` | 2-block CNN, lr=1e-3   | `result/stage_3_result/CIFAR/CNN_baseline/` |
| CIFAR B            | `CIFAR_lr_3e4`   | lr=3e-4                | `result/stage_3_result/CIFAR/CNN_lr_3e4/`   |
| CIFAR C            | `CIFAR_deeper`   | 3 conv blocks, lr=1e-3 | `result/stage_3_result/CIFAR/CNN_deeper/`   |


Examples:

```bash
PYTHONPATH=. python -u script/stage_3_script/script_cnn.py --preset MNIST_lr_3e4
PYTHONPATH=. python -u script/stage_3_script/script_cnn.py --preset ORL_deeper
PYTHONPATH=. python -u script/stage_3_script/script_cnn.py --preset CIFAR_deeper
```

## 3. Evaluation (what the script prints while running)

- After **Start**, you see which **preset** and **output folder** are used, then a one-line **setup summary**.
- **Loading:** `loading data...` and tensor shapes for train/test.
- **Training:** `Epoch: … train loss: …` about **every 5 epochs** (plus the last epoch).
- **Testing:** `--start testing...`, then **saving results...**, path to **learning_curve.png**, **evaluating performance...**, and the metric block (accuracy, macro/weighted/micro P/R/F1).
- **Finish** marks the end of that preset.

There is **no tqdm progress bar**; long stretches between epoch prints are normal when each epoch does many batches (especially CIFAR).

## 4. Saved outputs

Each preset writes to its own subfolder under `result/stage_3_result/`:


| File                  | What it is                                          |
| --------------------- | --------------------------------------------------- |
| `prediction_result_0` | Pickle of run outputs (predy, truey, learningcurve) |
| `learning_curve.png`  | Training loss per epoch                             |


