'''
Load pre-split Stage 3 image datasets (MNIST, ORL, CIFAR) from instructor pickle files.
'''

from code.base_class.dataset import dataset
import pickle
import numpy as np


class Dataset_Loader(dataset):
    dataset_source_folder_path = None
    # dataset_source_file_name should be 'MNIST', 'ORL', or 'CIFAR' (exact filename)
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _process_instances(self, instances, dataset_name):
        X, y = [], []
        for inst in instances:
            img = np.array(inst['image'], dtype=np.float32)
            label = inst['label']

            # Normalise to [0, 1] if stored as uint8
            if img.max() > 1.0:
                img = img / 255.0

            if dataset_name == 'MNIST':
                # img: (28, 28)  →  (1, 28, 28)
                img = img[np.newaxis, :, :]
            elif dataset_name == 'ORL':
                # img: (112, 92, 3) — grayscale stored as 3-channel; use R channel
                img = img[:, :, 0]          # (112, 92)
                img = img[np.newaxis, :, :] # (1, 112, 92)
                label = label - 1           # shift labels 1-40 → 0-39
            elif dataset_name == 'CIFAR':
                # img: (32, 32, 3)  →  (3, 32, 32)
                img = img.transpose(2, 0, 1)

            X.append(img)
            y.append(label)

        return np.stack(X, axis=0), np.array(y, dtype=np.int64)

    def load(self):
        print('loading data...', flush=True)
        path = self.dataset_source_folder_path + self.dataset_source_file_name
        with open(path, 'rb') as f:
            data = pickle.load(f)

        dataset_name = self.dataset_source_file_name  # 'MNIST', 'ORL', or 'CIFAR'
        X_train, y_train = self._process_instances(data['train'], dataset_name)
        X_test, y_test = self._process_instances(data['test'], dataset_name)
        print(f'{dataset_name} — train: {X_train.shape}, test: {X_test.shape}', flush=True)
        return {
            'train': {'X': X_train, 'y': y_train},
            'test':  {'X': X_test,  'y': y_test},
        }
