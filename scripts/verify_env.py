"""Quick import check for the course toolchain (PyTorch + scikit-learn + common libs)."""

from __future__ import annotations

import sys


def main() -> None:
    import numpy as np
    import pandas as pd
    import sklearn
    import torch
    import torchvision
    from matplotlib import pyplot as plt

    x = torch.randn(2, 3)
    _ = torch.nn.functional.relu(x)

    print("Python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("numpy:", np.__version__)
    print("pandas:", pd.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("matplotlib:", plt.matplotlib.__version__)
    print("OK — imports and a tiny torch op succeeded.")


if __name__ == "__main__":
    main()
