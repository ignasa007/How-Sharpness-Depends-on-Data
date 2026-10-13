from os.path import basename, splitext

import numpy as np
from torchvision import datasets
import matplotlib.pyplot as plt

from utilities import compute_hvp, train_gd
from constants import *


Dataset = datasets.CIFAR10
OUTPUT_DIM = 10
MAX_PROP = None
WIDTHS = (WIDE_SHALLOW_WIDTHS, NARROW_DEEP_WIDTHS)
LR = 5e-4


def compute_numerical_eigvals(X, Y, model, device):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sum_samples = 0.
    for i in range(1, HUTCHINSON_SAMPLES+1):
        random_vector = torch.randn(num_params, device=device)
        sum_samples += random_vector @ compute_hvp(X, Y, model, random_vector, device)
    sum_eigenvalues = sum_samples / HUTCHINSON_SAMPLES
    return sum_eigenvalues.item()

@torch.no_grad()
def compute_theoretical_eigvals(X, Y, model, device, n_eigs=1):
    W1, W2 = model._all_layers()
    X_spec_norm = torch.linalg.matrix_norm(X, 2)
    W1_svdvals_sq = torch.linalg.svdvals(W1.weight.data).square()
    W2_svdvals_sq = torch.linalg.svdvals(W2.weight.data).square()
    upper_bounds = X_spec_norm**2 / X.size(0) * (
        W1_svdvals_sq[:n_eigs, None] + W2_svdvals_sq[None, :n_eigs]
    ).flatten()
    upper_bounds = torch.sort(upper_bounds, descending=True).values[:n_eigs].to("cpu").numpy()
    return upper_bounds

fig, axs = plt.subplots(1, len(WIDTHS), figsize=(AX_WIDTH*len(WIDTHS), AX_HEIGHT*1))

for widths, ax in zip(WIDTHS, axs):

    _, out = train_gd(
        Dataset=Dataset, output_dim=OUTPUT_DIM, subset_size=SUBSET_SIZE,
        sigma_x=SIGMA_X, max_prop=MAX_PROP,
        Model=MODEL, model_kwargs=MODEL_KWARGS, widths=widths, balanced_init=False,
        act_name=ACTIVATION, act_kwargs=ACT_KWARGS,
        loss_type=LOSS_TYPE, lr=LR, sgd_kwargs=SGD_KWARGS,
        epochs=EPOCHS, log_every=LOG_EVERY, device=DEVICE, seed=SEED,
        tracker_fns=(compute_numerical_eigvals,),
    )

    _, param_norms, _, numerical_eigvals = out
    p = ax.plot(
        LOG_EVERY*np.arange(0, len(numerical_eigvals)),
        numerical_eigvals,
        linestyle="-", linewidth=5, alpha=1.0, color="blue"
    )
    ax.tick_params(axis="both", labelsize=15)
    ax.set_xlabel("Epoch", fontsize=18)
    ax.set_ylabel("Sharpness", fontsize=18, color=p[-1].get_color())
    ax.grid()
    ax_twin = ax.twinx()
    p = ax_twin.plot(
        np.arange(0, len(param_norms)),
        param_norms.sum(axis=1),
        linestyle="--", linewidth=5, alpha=1.0, color="brown"
    )
    ax_twin.tick_params(axis="both", labelsize=15)
    ax_twin.set_ylabel("Parameter Norm", fontsize=18, color=p[-1].get_color(), rotation=270, labelpad=20)

fig.tight_layout()
plt.savefig(f"{ASSETS}/{splitext(basename(__file__))[0]}-LR={LR}.png")
plt.close()