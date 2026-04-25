import argparse
import os

import numpy as np
import torch

from code.stage_1_code.Result_Saver import Result_Saver
from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Evaluate_Multiclass import Evaluate_Multiclass
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Setting_Prepartitioned import Setting_Prepartitioned

# Preset id -> (result subfolder, Method_MLP kwargs)
PRESETS = {
    'baseline': (
        'MLP_baseline',
        dict(hidden_dim=256, learning_rate=1e-3),
    ),
    'hidden_128': (
        'MLP_hidden_128',
        dict(hidden_dim=128, learning_rate=1e-3),
    ),
    'hidden_512': (
        'MLP_hidden_512',
        dict(hidden_dim=512, learning_rate=1e-3),
    ),
    'lr_3e4': (
        'MLP_lr_3e4',
        dict(hidden_dim=256, learning_rate=3e-4),
    ),
}

# Stage 2: MLP
if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Stage 2 MLP: train on train.csv, evaluate on test.csv. '
        'Use --preset to run baseline or ablation variants (Section 3.6–3.7).'
    )
    p.add_argument(
        '--preset',
        choices=list(PRESETS.keys()),
        default='baseline',
        help='Model variant: baseline (report default), or ablation presets.',
    )
    args = p.parse_args()

    result_subdir, mlp_kwargs = PRESETS[args.preset]
    out_dir = os.path.join('../../result/stage_2_result', result_subdir)
    os.makedirs(out_dir, exist_ok=True)
    if not out_dir.endswith(os.sep):
        out_dir = out_dir + os.sep

    np.random.seed(2)
    torch.manual_seed(2)

    data_obj = Dataset_Loader('mnist_csv', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.train_file_name = 'train.csv'
    data_obj.test_file_name = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron', '', **mlp_kwargs)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = out_dir
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Prepartitioned('prepartitioned train/test', '')
    evaluate_obj = Evaluate_Multiclass('multiclass metrics', '')

    print('************ Start ************')
    print('preset:', args.preset, '->', out_dir)
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    metrics, _ = setting_obj.load_run_save_evaluate()
    print('************ Test metrics (dict) ************')
    print(metrics)
    print('************ Finish ************')
