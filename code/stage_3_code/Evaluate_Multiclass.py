'''
Multiclass classification metrics: accuracy + precision/recall/F1
with macro, weighted, and micro averages.
'''

from code.base_class.evaluate import evaluate
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


class Evaluate_Multiclass(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...', flush=True)
        y_true = np.asarray(self.data['true_y']).ravel()
        y_pred = np.asarray(self.data['pred_y']).ravel()
        metrics = {'accuracy': accuracy_score(y_true, y_pred)}
        for avg in ('macro', 'weighted', 'micro'):
            metrics[f'precision_{avg}'] = precision_score(
                y_true, y_pred, average=avg, zero_division=0
            )
            metrics[f'recall_{avg}'] = recall_score(
                y_true, y_pred, average=avg, zero_division=0
            )
            metrics[f'f1_{avg}'] = f1_score(
                y_true, y_pred, average=avg, zero_division=0
            )
        for k, v in sorted(metrics.items()):
            print(f'  {k}: {v:.4f}', flush=True)
        return metrics
