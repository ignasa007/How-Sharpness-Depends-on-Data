import os

import numpy as np
from torchvision import datasets
import matplotlib.pyplot as plt

from utilities import compute_hvp, train_gd
from constants import *


DATASETS = (datasets.MNIST, datasets.FashionMNIST, datasets.CIFAR10)
OUTPUT_DIMS = (10, 10, 10)
MAX_PROP = None
WIDTHS = NARROW_DEEP_WIDTHS
LR = NARROW_DEEP_LR


def compute_numerical_eigvals(X, Y, model, device):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sum_samples = 0.
    for _ in range(1, HUTCHINSON_SAMPLES+1):
        random_vector = torch.randn(num_params, device=device)
        sum_samples += random_vector @ compute_hvp(X, Y, model, random_vector, device)
    sum_eigenvalues = sum_samples / HUTCHINSON_SAMPLES
    return sum_eigenvalues.item()

@torch.no_grad()
def compute_theoretical_eigvals(X, Y, model, device):
    all_layers = model._all_layers()
    prefix_fro_norms = []
    prod = torch.eye(all_layers[0].in_features, device=device)
    for layer in all_layers:
        prefix_fro_norms.append(torch.linalg.norm(prod, ord="fro").item())     # Frobenius norm
        prod = layer.weight.data @ prod
    suffix_fro_norms = []
    prod = torch.eye(all_layers[-1].out_features, device=device)
    for layer in all_layers[::-1]:
        suffix_fro_norms.append(torch.linalg.norm(prod, ord="fro").item())     # Frobenius norm
        prod = prod @ layer.weight.data
    return (np.array(prefix_fro_norms)**2 * np.array(suffix_fro_norms[::-1])**2).sum()

fig, axs = plt.subplots(1, len(DATASETS), figsize=(AX_WIDTH*len(DATASETS), AX_HEIGHT*1))

for Dataset, output_dim, ax in zip(DATASETS, OUTPUT_DIMS, axs):

    _, out = train_gd(
        Dataset=Dataset, output_dim=output_dim, subset_size=SUBSET_SIZE,
        sigma_x=SIGMA_X, max_prop=MAX_PROP,
        Model=MODEL, model_kwargs=MODEL_KWARGS, widths=WIDTHS, balanced_init=False,
        act_name=ACTIVATION, act_kwargs=ACT_KWARGS,
        loss_type=LOSS_TYPE, lr=LR, sgd_kwargs=SGD_KWARGS,
        epochs=EPOCHS, log_every=LOG_EVERY, device=DEVICE, seed=SEED,
        tracker_fns=(compute_numerical_eigvals, compute_theoretical_eigvals)
    )

    losses, *_, numerical_eigvals, theoretical_eigvals = out

    ax.plot(
        LOG_EVERY*np.arange(0, len(numerical_eigvals)),
        numerical_eigvals,
        linewidth=5, linestyle="-", color="green", alpha=1.0,
        label=r"Tr$(\nabla^2\mathcal{L})$ (Numerical)"
    )
    ax.plot(
        LOG_EVERY*np.arange(0, len(theoretical_eigvals)),
        SIGMA_X**2 * theoretical_eigvals,
        linewidth=5, linestyle="--", color="magenta", alpha=1.0,
        label=r"Tr$(\nabla^2\mathcal{L})$ (Theoretical)"
    )

    ax.tick_params(axis="both", labelsize=15)
    ax.set_xlabel("Epoch", fontsize=18)
    ax.set_ylabel("Sharpness", fontsize=18)
    ax.grid()

    ax_twin = ax.twinx()
    ax_twin.plot(losses, color="gray", linewidth=5, alpha=1.0)
    ax_twin.set_ylabel(fontsize=18, rotation=270, labelpad=20)

axs[0].twinx().legend(*ax.get_legend_handles_labels(), loc="upper left", fontsize=18, framealpha=1.0)

fig.tight_layout()
fn = f"{ASSETS}/{os.path.splitext(os.path.basename(__file__))[0]}-LR={LR}.png"
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn)
plt.close()
print(f"\nPlot saved at {fn}")