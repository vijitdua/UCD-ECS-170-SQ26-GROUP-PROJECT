'''
Load pre-split Stage 2 CSV
'''

from code.base_class.dataset import dataset
import numpy as np


class Dataset_Loader(dataset):
    dataset_source_folder_path = None
    train_file_name = 'train.csv'
    test_file_name = 'test.csv'

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    # Read the CSV file and return the data (X) and labels (y)
    def _read_csv(self, path):
        X, y = [], []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [int(x) for x in line.split(',')]
                y.append(parts[0])
                X.append(parts[1:])
        X = np.asarray(X, dtype=np.float32) / 255.0
        y = np.asarray(y, dtype=np.int64)
        return X, y

    # Load the data from the CSV files and return the data (X) and labels (y)
    def load(self):
        print('loading data...')
        folder = self.dataset_source_folder_path
        X_train, y_train = self._read_csv(folder + self.train_file_name)
        X_test, y_test = self._read_csv(folder + self.test_file_name)
        print('train:', X_train.shape, 'test:', X_test.shape)
        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test},
        }
