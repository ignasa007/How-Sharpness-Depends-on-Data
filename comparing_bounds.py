from os.path import basename, splitext
from functools import partial

import numpy as np
from torch.nn.utils import parameters_to_vector
from torchvision import datasets
import matplotlib.pyplot as plt

from utilities import MLP, compute_hvp, lanczos, train_gd
from constants import *


Datasets = (datasets.MNIST, datasets.FashionMNIST, datasets.CIFAR10)
OUTPUT_DIM, SIGMA_X, MAX_PROP = 10, 1., None
MODEL, WIDTHS, MODEL_KWARGS = MLP, [256]*1, dict()
ACTIVATION, ACT_KWARGS = "Linear", dict()
LOSS_TYPE, LR, OPTIM_KWARGS = "Mean Squared Error", 1e-3, dict()
N_EIGS = 1


def compute_numerical_eigvals(X, Y, model, device, n_eigs=1):
    model.zero_grad()
    parameters = parameters_to_vector(model.parameters()).detach().cpu()
    hvp_delta = lambda vector: compute_hvp(X, Y, model, vector, device).detach().cpu()
    eigenvalues, _ = lanczos(tensor_matvec=hvp_delta, p=len(parameters), n_eigs=n_eigs)
    return eigenvalues.numpy()

@torch.no_grad()
def compute_theoretical_eigvals(X, Y, model, device, n_eigs=1):
    W1, W2 = model._all_layers()
    W1_spec_norm = torch.linalg.matrix_norm(W1.weight.data, 2)
    W2_spec_norm = torch.linalg.matrix_norm(W2.weight.data, 2)
    return (W1_spec_norm**2 + W2_spec_norm**2).cpu().numpy(), torch.max(W1_spec_norm**2, W2_spec_norm**2).cpu().numpy()

fig, axs = plt.subplots(1, len(Datasets), figsize=(7.2*len(Datasets), 4.8*1))

for Dataset, ax in zip(Datasets, axs):

    _, out = train_gd(
        Dataset=Dataset, output_dim=OUTPUT_DIM, subset_size=SUBSET_SIZE,
        sigma_x=SIGMA_X, max_prop=MAX_PROP,
        Model=MODEL, model_kwargs=MODEL_KWARGS, widths=WIDTHS, balanced_init=False,
        act_name=ACTIVATION, act_kwargs=ACT_KWARGS,
        loss_type=LOSS_TYPE, lr=LR, optim_kwargs=OPTIM_KWARGS,
        epochs=EPOCHS, log_every=LOG_EVERY, device=DEVICE, seed=SEED,
        tracker_fns=(
            partial(compute_numerical_eigvals, n_eigs=N_EIGS),
            partial(compute_theoretical_eigvals, n_eigs=N_EIGS),
        )
    )

    *_, numerical_eigvals, theoretical_eigvals = out

    ax.plot(
        LOG_EVERY*np.arange(0, len(numerical_eigvals)),
        numerical_eigvals,
        linewidth=5, color="black", alpha=1.0, label="Sharpness"
    )
    ax.plot(
        LOG_EVERY*np.arange(0, len(theoretical_eigvals)),
        theoretical_eigvals[:,0],
        linewidth=5, linestyle="--", color="blue", alpha=1.0,
        label=r"$\|W_1\|_2^2+\|W_2\|_2^2$"
    )
    ax.plot(
        LOG_EVERY*np.arange(0, len(theoretical_eigvals)),
        theoretical_eigvals[:,1],
        linewidth=5, linestyle="--", color="brown", alpha=1.0,
        label=r"$\max\{\|W_1\|_2^2, \|W_2\|_2^2\}$"
    )

    ax.tick_params(axis="both", labelsize=15)
    ax.set_xlabel("Epoch", fontsize=18)
    ax.grid()

axs[0].legend(loc="upper right", fontsize=18, framealpha=1.0)

fig.tight_layout()
plt.savefig(f"{ASSETS}/{splitext(basename(__file__))[0]}-LR={LR}.png")
plt.close()