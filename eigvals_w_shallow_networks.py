from os.path import basename, splitext
from functools import partial
from itertools import cycle

import numpy as np
from torch.nn.utils import parameters_to_vector
from torchvision import datasets
import matplotlib.pyplot as plt

from utilities import compute_hvp, lanczos, train_gd
from constants import *


DATASETS = (datasets.MNIST, datasets.FashionMNIST, datasets.CIFAR10)
OUTPUT_DIMS = (10, 10, 10)
MAX_PROP = None
WIDTHS = WIDE_SHALLOW_WIDTHS
LR = WIDE_SHALLOW_LR
N_EIGS = 10


def compute_numerical_eigvals(X, Y, model, device, n_eigs=1):
    model.zero_grad()
    parameters = parameters_to_vector(model.parameters()).detach().cpu()
    hvp_delta = lambda vector: compute_hvp(X, Y, model, vector, device).detach().cpu()
    eigenvalues, _ = lanczos(tensor_matvec=hvp_delta, p=len(parameters), n_eigs=n_eigs)
    return eigenvalues.numpy()

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

fig, axs = plt.subplots(1, len(DATASETS), figsize=(AX_WIDTH*len(DATASETS), AX_HEIGHT*1))

for Dataset, output_dim, ax in zip(DATASETS, OUTPUT_DIMS, axs):

    _, out = train_gd(
        Dataset=Dataset, output_dim=output_dim, subset_size=SUBSET_SIZE,
        sigma_x=SIGMA_X, max_prop=MAX_PROP,
        Model=MODEL, model_kwargs=MODEL_KWARGS, widths=WIDTHS, balanced_init=False,
        act_name=ACTIVATION, act_kwargs=ACT_KWARGS,
        loss_type=LOSS_TYPE, lr=LR, sgd_kwargs=SGD_KWARGS,
        epochs=EPOCHS, log_every=LOG_EVERY, device=DEVICE, seed=SEED,
        tracker_fns=(
            partial(compute_numerical_eigvals, n_eigs=N_EIGS),
            partial(compute_theoretical_eigvals, n_eigs=N_EIGS),
        )
    )

    *_, numerical_eigvals, theoretical_eigvals = out

    linestyles = cycle(["-", "--"])
    for i in range(N_EIGS):
        ax.plot(
            LOG_EVERY*np.arange(0, len(numerical_eigvals)),
            100*np.abs(1-theoretical_eigvals[:,i]/numerical_eigvals[:,i]),
            linewidth=5, linestyle=next(linestyles), alpha=1.0, label=rf"$\lambda_{{{i+1}}}$"
        )

    ax.tick_params(axis="both", labelsize=15)
    ax.set_xlabel("Epoch", fontsize=18)
    ax.grid()

axs[0].set_ylabel(r"%Error in $\lambda_{{i}}$ Approximation", fontsize=18)
axs[0].legend(loc="upper right", ncol=2, fontsize=18, framealpha=1.0)

fig.tight_layout()
plt.savefig(f"{ASSETS}/{splitext(basename(__file__))[0]}-LR={LR}.png")
plt.close()