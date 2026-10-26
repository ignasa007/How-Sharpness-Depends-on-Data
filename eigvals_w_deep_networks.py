import os
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
MAX_PROP = 0.1
WIDTHS = NARROW_DEEP_WIDTHS
LR = NARROW_DEEP_LR
N_EIGS = 10


def compute_numerical_eigvals(X, Y, model, device, n_eigs=1):
    model.zero_grad()
    parameters = parameters_to_vector(model.parameters()).detach().cpu()
    hvp_delta = lambda vector: compute_hvp(X, Y, model, vector, device).detach().cpu()
    eigenvalues, _ = lanczos(tensor_matvec=hvp_delta, p=len(parameters), n_eigs=n_eigs)
    return eigenvalues.numpy()

@torch.no_grad()
def compute_theoretical_eigvals(X, Y, model, device, n_eigs=1):
    L = len(list(model.parameters()))
    sigma_2L = (torch.linalg.svdvals(Y) / torch.linalg.matrix_norm(X, 2)).square()
    sigma_2 = sigma_2L ** (1/L)
    num = sigma_2L.unsqueeze(1) - sigma_2L.unsqueeze(0)
    den = sigma_2.unsqueeze(1) - sigma_2.unsqueeze(0)
    result = torch.linalg.norm(X, ord=2).square() / X.size(0) \
        * torch.where(den.abs()>1e-8, num/den, (L*sigma_2L/sigma_2).unsqueeze(1))
    return torch.sort(result, descending=True).values.to("cpu").flatten()[:n_eigs].numpy()

fig, axs = plt.subplots(1, len(DATASETS), figsize=(AX_WIDTH*len(DATASETS), AX_HEIGHT*1))

for Dataset, output_dim, ax in zip(DATASETS, OUTPUT_DIMS, axs):

    _, out = train_gd(
        Dataset=Dataset, output_dim=output_dim, subset_size=SUBSET_SIZE,
        sigma_x=SIGMA_X, max_prop=MAX_PROP,
        Model=MODEL, model_kwargs=MODEL_KWARGS, widths=WIDTHS, balanced_init=True,
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
            numerical_eigvals[:,i],
            linewidth=5, linestyle=next(linestyles), alpha=1.0, label=rf"$\lambda_{{{i+1}}}$"
        )
    ax.axhline(
        theoretical_eigvals[0,0], 0., LOG_EVERY*(len(theoretical_eigvals)-1),
        linewidth=5, linestyle="--", color="black", alpha=1.0
    )

    ax.tick_params(axis="both", labelsize=15)
    ax.set_xlabel("Epoch", fontsize=18)
    ax.grid()

axs[0].set_ylabel(r"Eigenvalue", fontsize=18)
axs[0].legend(loc="upper right", ncol=2, fontsize=18, framealpha=1.0)

fig.tight_layout()
fn = f"{ASSETS}/{os.path.splitext(os.path.basename(__file__))[0]}-LR={LR}.png"
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn)
plt.close()
print(f"\nPlot saved at {fn}")