'''
First, we show that retaining the right singular vectors of the design matrix, violating Assumption 2 in the paper,
does not change anything since these can be absorbed into a reparameterization of the first layer. Therefore, that
assumption can be dropped without changing any of the results.

**Note:** That being said, the singular values must be well-behaved.
'''


import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from utilities import loss_fn
from constants import *


DATASETS = (datasets.MNIST, datasets.FashionMNIST, datasets.CIFAR10)
OUTPUT_DIMS = (10, 10, 10)
fig, ax = plt.subplots(1, 1, figsize=(AX_WIDTH*1, AX_HEIGHT*1))
width = 0.8


original_losses, after_projecting = list(), list()
for Dataset, output_dim in zip(DATASETS, OUTPUT_DIMS):
    full_trainset = Dataset(
        root=f"{DATASTORE}/{Dataset.__name__.lower()}",
        train=True, download=True, transform=transforms.ToTensor()
    )
    data_loader = torch.utils.data.DataLoader(full_trainset, batch_size=SUBSET_SIZE)
    x, y = next(iter(data_loader))
    # Note that we work with transposed coordinates, in comparison with the theory
    x = x.view(x.size(0), -1).float().to(DEVICE)                # (n, d_0)
    y = F.one_hot(y, num_classes=output_dim).float().to(DEVICE) # (n, d_L)
    w = torch.linalg.pinv(x) @ y
    loss = loss_fn(x@w, y, loss_type="Mean Squared Error")
    original_losses.append(loss.item())
    _, r = torch.linalg.qr(x.T, mode="r")
    x = r.T
    w = torch.linalg.pinv(x) @ y
    loss = loss_fn(x@w, y, loss_type="Mean Squared Error")
    after_projecting.append(loss.item())

ax.bar(np.arange(len(DATASETS))-width/4, original_losses, width=width/2, label="Original Dataset")
ax.bar(np.arange(len(DATASETS))+width/4, after_projecting, width=width/2, label="Projected onto PCs")
ax.set_xticks(ticks=np.arange(len(DATASETS)), labels=list(map(lambda x: x.__name__, DATASETS)), fontsize=18)
ax.tick_params(axis="y", labelsize=15)
ax.set_ylabel("Optimal Loss", fontsize=18)
ax.legend(loc="upper left", fontsize=18, framealpha=1.0)
ax.grid()

fig.tight_layout()
fn = f"{ASSETS}/{os.path.splitext(os.path.basename(__file__))[0]}.png"
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn)
plt.close()
print(f"\nPlot saved at {fn}")