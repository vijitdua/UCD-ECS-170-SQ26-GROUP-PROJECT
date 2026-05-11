'''
Configurable CNN for image classification (Stage 3).

Architecture: two Conv-BN-ReLU-MaxPool blocks → AdaptiveAvgPool(4×4) → FC head.
AdaptiveAvgPool makes the classifier shape-agnostic, so the same class handles
all three datasets (MNIST 28×28, ORL 112×92, CIFAR 32×32) without change.
'''

from code.base_class.method import method
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _training_device():
    """Prefer NVIDIA CUDA, then Apple Metal (MPS), else CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _describe_training_device(device):
    """Human-readable label for logging."""
    if device.type == 'cuda':
        try:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            return f'CUDA ({name})'
        except Exception:
            return 'CUDA'
    if device.type == 'mps':
        return 'MPS (Apple Metal)'
    return 'CPU'


class Method_CNN(method, nn.Module):
    data = None

    # Training hyper-parameters (overridable via kwargs)
    max_epoch = 30
    learning_rate = 1e-3
    batch_size = 64

    # Network shape (overridable via kwargs)
    in_channels = 1
    num_classes = 10
    num_conv_blocks = 2   # 2 = baseline; 3 = deeper ablation

    def __init__(self, mName, mDescription, **kwargs):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        for k in ('max_epoch', 'learning_rate', 'batch_size',
                  'in_channels', 'num_classes', 'num_conv_blocks'):
            if k in kwargs and kwargs[k] is not None:
                setattr(self, k, kwargs[k])

        self._device = _training_device()
        print(
            f'Method_CNN compute device: {self._device} ({_describe_training_device(self._device)})',
            flush=True,
        )
        self._learning_curve = {'epoch': [], 'train_loss': []}

        # Build conv blocks dynamically (2 or 3)
        layers = [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        out_ch = 64
        if self.num_conv_blocks >= 3:
            layers += [
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            out_ch = 128

        self.features = nn.Sequential(*layers)

        # Collapse spatial dims to 4×4 regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Linear(out_ch * 4 * 4, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.contiguous()
        # MPS: adaptive average pooling is implemented only for divisibility patterns
        # many feature-map sizes violate (#96056). PyTorch's F.interpolate(..., mode='area')
        # also lowers to adaptive_avg_pool2d for downsampling, so it fails too (e.g. 7×7→4×4).
        # Run pooling on CPU and move back; keep batches contiguous (fit/test) so backward
        # through the classifier stays stable on MPS.
        if x.device.type == 'mps':
            x = F.adaptive_avg_pool2d(x.cpu(), (4, 4)).to(
                device=x.device, dtype=x.dtype
            ).contiguous()
        else:
            x = self.adaptive_pool(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)

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
                idx = perm[start:start + self.batch_size]
                xb = torch.from_numpy(np.ascontiguousarray(X_train[idx])).to(
                    device=self._device, dtype=torch.float32, non_blocking=False
                )
                yb = torch.as_tensor(y_train[idx], dtype=torch.long, device=self._device)
                optimizer.zero_grad()
                loss = loss_fn(self.forward(xb), yb)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            mean_loss = float(np.mean(losses))
            self._learning_curve['epoch'].append(epoch)
            self._learning_curve['train_loss'].append(mean_loss)
            if epoch % 5 == 0 or epoch == self.max_epoch - 1:
                print(f'Epoch: {epoch:3d}  train loss: {mean_loss:.4f}', flush=True)

    def test(self, X):
        self.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                xb = torch.from_numpy(
                    np.ascontiguousarray(X[start:start + self.batch_size])
                ).to(device=self._device, dtype=torch.float32, non_blocking=False)
                preds.append(self.forward(xb).argmax(dim=1).cpu())
        return torch.cat(preds).numpy()

    def run(self):
        print('method running...', flush=True)
        print('--start training...', flush=True)
        self.fit(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...', flush=True)
        pred_y = self.test(self.data['test']['X'])
        true_y = np.asarray(self.data['test']['y'])
        return {
            'pred_y': pred_y,
            'true_y': true_y,
            'learning_curve': self._learning_curve,
        }
