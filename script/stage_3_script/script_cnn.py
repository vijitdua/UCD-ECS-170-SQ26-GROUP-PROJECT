'''
Stage 3: Image classification with CNN.

Run from repo root — **all presets in one shot**:

    PYTHONPATH=. .venv/bin/python -u script/stage_3_script/script_cnn.py --preset all

Single preset example:

    PYTHONPATH=. .venv/bin/python -u script/stage_3_script/script_cnn.py --preset MNIST_baseline

``--preset all`` runs every preset **one after another** (MNIST → ORL → CIFAR,
each baseline then lr ablation then deeper). Progress prints every few epochs
during training; pickling and metric lines print at the end of each run.

Available presets
-----------------
Baseline runs (one per required dataset):
  MNIST_baseline   — MNIST digits, lr=1e-3
  ORL_baseline     — ORL faces (1-ch grayscale, 40 classes), lr=1e-3
  CIFAR_baseline   — CIFAR-10 colored objects (3-ch), lr=1e-3

Ablation variants (for Section 3-5 of the report):
  MNIST_lr_3e4     — MNIST, lr=3e-4
  CIFAR_lr_3e4     — CIFAR-10, lr=3e-4
  ORL_lr_3e4       — ORL, lr=3e-4

Results are saved under ``result/stage_3_result/<dataset>/<variant>/``:

  prediction_result_0   — pickle with pred_y, true_y, learning_curve
  learning_curve.png    — epoch vs mean train loss
'''

import argparse
import os
import sys
import traceback
from pathlib import Path

# NumPy/PyTorch are heavy; the process can sit silent for tens of seconds here on
# first import — tell the user immediately so the terminal doesn't look hung.
print(
    'stage_3 script: importing NumPy and PyTorch (first time after boot can take a while)...',
    file=sys.stderr,
    flush=True,
)
import numpy as np
import torch

print(f'stage_3 script: PyTorch {torch.__version__} loaded.', file=sys.stderr, flush=True)

# Resolve repo root from this file's location so the script works regardless
# of which directory it is invoked from.
_SCRIPT_DIR = Path(__file__).resolve().parent          # script/stage_3_script/
_REPO_ROOT  = _SCRIPT_DIR.parent.parent                # repo root
_DATA_DIR   = _REPO_ROOT / 'data'  / 'stage_3_data'
_RESULT_DIR = _REPO_ROOT / 'result' / 'stage_3_result'

from code.stage_1_code.Result_Saver import Result_Saver
from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Evaluate_Multiclass import Evaluate_Multiclass
from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Setting_Prepartitioned import Setting_Prepartitioned

# ---------------------------------------------------------------------------
# Preset registry: dataset file name + CNN kwargs + result sub-directory
# ---------------------------------------------------------------------------
PRESETS = {
    # ---- MNIST (28×28 grayscale, 10 classes, 60k/10k) --------------------
    'MNIST_baseline': dict(
        dataset='MNIST',
        result_subdir='MNIST/CNN_baseline',
        cnn_kwargs=dict(in_channels=1, num_classes=10,
                        max_epoch=30, learning_rate=1e-3, batch_size=64),
    ),
    'MNIST_lr_3e4': dict(
        dataset='MNIST',
        result_subdir='MNIST/CNN_lr_3e4',
        cnn_kwargs=dict(in_channels=1, num_classes=10,
                        max_epoch=30, learning_rate=3e-4, batch_size=64),
    ),

    # ---- ORL faces (112×92, 1-ch grayscale, 40 classes, 360/40) ----------
    'ORL_baseline': dict(
        dataset='ORL',
        result_subdir='ORL/CNN_baseline',
        cnn_kwargs=dict(in_channels=1, num_classes=40,
                        max_epoch=100, learning_rate=1e-3, batch_size=32),
    ),
    'ORL_lr_3e4': dict(
        dataset='ORL',
        result_subdir='ORL/CNN_lr_3e4',
        cnn_kwargs=dict(in_channels=1, num_classes=40,
                        max_epoch=100, learning_rate=3e-4, batch_size=32),
    ),

    # ---- CIFAR-10 (32×32 RGB, 10 classes, 50k/10k) -----------------------
    'CIFAR_baseline': dict(
        dataset='CIFAR',
        result_subdir='CIFAR/CNN_baseline',
        cnn_kwargs=dict(in_channels=3, num_classes=10,
                        max_epoch=50, learning_rate=1e-3, batch_size=128),
    ),
    'CIFAR_lr_3e4': dict(
        dataset='CIFAR',
        result_subdir='CIFAR/CNN_lr_3e4',
        cnn_kwargs=dict(in_channels=3, num_classes=10,
                        max_epoch=50, learning_rate=3e-4, batch_size=128),
    ),
    'CIFAR_deeper': dict(
        dataset='CIFAR',
        result_subdir='CIFAR/CNN_deeper',
        cnn_kwargs=dict(in_channels=3, num_classes=10, num_conv_blocks=3,
                        max_epoch=50, learning_rate=1e-3, batch_size=128),
    ),

    # ---- Deeper-CNN ablations for MNIST and ORL --------------------------
    'MNIST_deeper': dict(
        dataset='MNIST',
        result_subdir='MNIST/CNN_deeper',
        cnn_kwargs=dict(in_channels=1, num_classes=10, num_conv_blocks=3,
                        max_epoch=30, learning_rate=1e-3, batch_size=64),
    ),
    'ORL_deeper': dict(
        dataset='ORL',
        result_subdir='ORL/CNN_deeper',
        cnn_kwargs=dict(in_channels=1, num_classes=40, num_conv_blocks=3,
                        max_epoch=100, learning_rate=1e-3, batch_size=32),
    ),
}

# Order for ``--preset all`` (every key in PRESETS appears exactly once).
PRESET_ORDER = [
    'MNIST_baseline',
    'MNIST_lr_3e4',
    'MNIST_deeper',
    'ORL_baseline',
    'ORL_lr_3e4',
    'ORL_deeper',
    'CIFAR_baseline',
    'CIFAR_lr_3e4',
    'CIFAR_deeper',
]


def _configure_stdout():
    """Line-buffer stdout/stderr so progress lines show up immediately."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass


def run_single_preset(preset_name: str):
    """Train + evaluate one preset; print progress to stdout (flush each line)."""
    cfg = PRESETS[preset_name]
    out_dir = _RESULT_DIR / cfg['result_subdir']
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_str = str(out_dir) + os.sep

    np.random.seed(2)
    torch.manual_seed(2)

    data_obj = Dataset_Loader(cfg['dataset'], '')
    data_obj.dataset_source_folder_path = str(_DATA_DIR) + os.sep
    data_obj.dataset_source_file_name = cfg['dataset']

    method_obj = Method_CNN('cnn', '', **cfg['cnn_kwargs'])

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = out_dir_str
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Prepartitioned('prepartitioned train/test', '')
    evaluate_obj = Evaluate_Multiclass('multiclass metrics', '')

    print('************ Start ************', flush=True)
    print(f'preset: {preset_name}  →  {out_dir_str}', flush=True)
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metrics, _ = setting_obj.load_run_save_evaluate()
    print('************ Test metrics ************', flush=True)
    print(metrics, flush=True)
    print('************ Finish ************', flush=True)
    return metrics


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    _configure_stdout()

    single_choices = list(PRESETS.keys())
    p = argparse.ArgumentParser(description='Stage 3 CNN — image classification')
    p.add_argument(
        '--preset',
        choices=single_choices + ['all'],
        default='MNIST_baseline',
        help='One preset, or "all" to run every preset sequentially (see PRESET_ORDER).',
    )
    args = p.parse_args()

    print(f'stage_3 script: CLI preset={args.preset!r} — starting runs.', file=sys.stderr, flush=True)

    if args.preset == 'all':
        missing = [k for k in PRESET_ORDER if k not in PRESETS]
        if missing:
            print('Internal error: PRESET_ORDER keys missing from PRESETS:', missing, flush=True)
            sys.exit(1)
        failed = []
        total = len(PRESET_ORDER)
        for i, name in enumerate(PRESET_ORDER, 1):
            sep = '=' * 72
            print(f'\n{sep}\n[{i}/{total}] NEXT: {name}\n{sep}\n', flush=True)
            try:
                run_single_preset(name)
            except Exception:
                print(f'\n*** FAILED: {name} ***\n', flush=True)
                traceback.print_exc()
                failed.append(name)
        if failed:
            print('\n************ SUMMARY: some presets failed ************', flush=True)
            print('Failed:', ', '.join(failed), flush=True)
            sys.exit(1)
        print('\n************ ALL PRESETS FINISHED OK ************', flush=True)
    else:
        run_single_preset(args.preset)
