'''
Train on the provided training split, evaluate on the provided test split.
Saves predictions (pickle) and a learning-curve PNG into the result folder.
'''

from code.base_class.setting import setting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Setting_Prepartitioned(setting):
    def load_run_save_evaluate(self):
        bundle = self.dataset.load()
        self.method.data = bundle
        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.fold_count = 0
        self.result.save()

        lc = learned_result.get('learning_curve') or {}
        losses = lc.get('train_loss') or []
        if losses:
            out_path = self.result.result_destination_folder_path + 'learning_curve.png'
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(lc.get('epoch', list(range(len(losses)))), losses)
            ax.set_xlabel('epoch')
            ax.set_ylabel('mean train loss')
            ax.set_title('Learning curve')
            fig.tight_layout()
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            print('saved learning curve to', out_path, flush=True)

        self.evaluate.data = {
            'true_y': learned_result['true_y'],
            'pred_y': learned_result['pred_y'],
        }
        return self.evaluate.evaluate(), None
