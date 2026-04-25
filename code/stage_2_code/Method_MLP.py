'''
MLP for flattened 784-dim inputs, 10-class logits + batched training.
'''

from code.base_class.method import method
import numpy as np
import torch
from torch import nn


class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 25
    learning_rate = 1e-3
    batch_size = 1024
    hidden_dim = 256

    def __init__(self, mName, mDescription, **kwargs):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        for k in ('max_epoch', 'learning_rate', 'batch_size', 'hidden_dim'):
            if k in kwargs and kwargs[k] is not None:
                setattr(self, k, kwargs[k])
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._learning_curve = {'epoch': [], 'train_loss': []}

        self.fc1 = nn.Linear(784, self.hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, 10)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

    def fit(self, X_train, y_train):
        self.to(self._device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        n = len(X_train)
        self._learning_curve = {'epoch': [], 'train_loss': []}

        for epoch in range(self.max_epoch):
            nn.Module.train(self, True)
            perm = np.random.permutation(n)
            losses = []
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = torch.as_tensor(X_train[idx], device=self._device)
                yb = torch.as_tensor(y_train[idx], dtype=torch.long, device=self._device)
                optimizer.zero_grad()
                logits = self.forward(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            mean_loss = float(np.mean(losses))
            self._learning_curve['epoch'].append(epoch)
            self._learning_curve['train_loss'].append(mean_loss)
            if epoch % 5 == 0 or epoch == self.max_epoch - 1:
                print('Epoch:', epoch, 'train loss:', round(mean_loss, 4))

    def test(self, X):
        self.eval()
        with torch.no_grad():
            xb = torch.as_tensor(X, device=self._device)
            logits = self.forward(xb)
            return logits.argmax(dim=1).cpu()

    def run(self):
        print('method running...')
        print('--start training...')
        self.fit(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X']).numpy()
        true_y = np.asarray(self.data['test']['y'])
        return {
            'pred_y': pred_y,
            'true_y': true_y,
            'learning_curve': self._learning_curve,
        }
