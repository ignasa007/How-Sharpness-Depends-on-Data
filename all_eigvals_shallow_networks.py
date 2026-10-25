import os
from functools import partial

import numpy as np
from torchvision import datasets
import matplotlib.pyplot as plt

from utilities import pad, train_gd
from constants import *


DATASETS = (datasets.MNIST, datasets.FashionMNIST)
OUTPUT_DIMS = (10, 10)
MAX_PROP = None
WIDTHS = [32]*1
LR = WIDE_SHALLOW_LR
N_EIGS = 10


def compute_numerical_eigvals(X, Y, model, device, n_eigs=None):

    weights = [p for p in model.parameters()]

    def get_loss(flat_params):
        out, pointer = X, 0
        for l in range(len(weights)):
            W = flat_params[pointer:pointer+weights[l].numel()].view_as(weights[l])
            out = out @ W.T
            pointer += weights[l].numel()
        return 0.5 / X.size(0) * torch.sum((out-Y)**2)

    flat_W = torch.cat([W.flatten() for W in weights])
    H = torch.autograd.functional.hessian(get_loss, flat_W)
    eigs_actual = torch.linalg.eigvalsh(H)
    return torch.sort(eigs_actual, descending=True).values.cpu().numpy()[:n_eigs]

@torch.no_grad()
def compute_theoretical_eigvals(X, Y, model, device, n_eigs=None):
    W1, W2 = model._all_layers()
    X_spec_norm = torch.linalg.matrix_norm(X, 2)
    # Can remove the padding and verify that the eigenvalues don't match
    W1_svdvals_sq = pad(torch.linalg.svdvals(W1.weight.data).square(), W1.weight.data.size(1))
    W2_svdvals_sq = pad(torch.linalg.svdvals(W2.weight.data).square(), W2.weight.data.size(0))
    upper_bounds = X_spec_norm**2 / X.size(0) * (
        W1_svdvals_sq[:n_eigs, None] + W2_svdvals_sq[None, :n_eigs]
    ).flatten().to("cpu").numpy()
    upper_bounds = np.sort(upper_bounds, descending=True)[:n_eigs]
    return upper_bounds

fig, axs = plt.subplots(1, len(DATASETS), figsize=(AX_WIDTH*len(DATASETS), AX_HEIGHT*1))
if not hasattr(axs, "__len__"):
    axs = np.array((axs,))

for Dataset, output_dim, ax in zip(DATASETS, OUTPUT_DIMS, axs):

    assets, _ = train_gd(
        Dataset=Dataset, output_dim=output_dim, subset_size=SUBSET_SIZE,
        sigma_x=SIGMA_X, max_prop=MAX_PROP,
        Model=MODEL, model_kwargs=MODEL_KWARGS, widths=WIDTHS, balanced_init=False,
        act_name=ACTIVATION, act_kwargs=ACT_KWARGS,
        loss_type=LOSS_TYPE, lr=LR, sgd_kwargs=SGD_KWARGS,
        epochs=EPOCHS, log_every=float("inf"), device=DEVICE, seed=SEED,
        tracker_fns=(
            partial(compute_numerical_eigvals, n_eigs=N_EIGS),
            partial(compute_theoretical_eigvals, n_eigs=N_EIGS),
        )
    )

    train_inputs, one_hot, model = assets
    numerical_eigvals = compute_numerical_eigvals(train_inputs, one_hot, model, DEVICE)
    theoretical_eigvals = compute_theoretical_eigvals(train_inputs, one_hot, model, DEVICE)
    theoretical_eigvals = theoretical_eigvals[theoretical_eigvals>0.]
    ax.plot(range(1, len(numerical_eigvals)+1), numerical_eigvals, linewidth=7.5, label="Numerical Eigenvalues")
    ax.plot(range(1, len(theoretical_eigvals)+1), theoretical_eigvals, linewidth=5, linestyle="--", label="Analytical Eigenvalues")

    ax.tick_params(axis="both", labelsize=15)
    ax.set_xlabel(r"Index, $i$", fontsize=18)
    ax.set_title(Dataset.__name__, fontsize=20)
    ax.set_xscale("log")
    ax.grid()

axs[0].legend(loc="lower left", fontsize=18, framealpha=1.0)

fig.tight_layout()
fn = f"{ASSETS}/{os.path.splitext(os.path.basename(__file__))[0]}.png"
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn)
plt.close()
print(f"\nPlot saved at {fn}")