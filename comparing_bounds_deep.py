import os
from functools import partial

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
from torchvision import datasets
import matplotlib.pyplot as plt

from utilities import compute_hvp, lanczos, train_gd
from constants import *


def compute_numerical_eigvals(inputs, labels, model, loss_type, device, n_eigs=1):
    model.zero_grad()
    parameters = parameters_to_vector(model.parameters()).detach().cpu()
    hvp_delta = lambda vector: compute_hvp(inputs, labels, model, loss_type, vector, device).detach().cpu()
    eigenvalues, _ = lanczos(tensor_matvec=hvp_delta, p=len(parameters), n_eigs=n_eigs)
    return eigenvalues.numpy()

@torch.no_grad()
def compute_theoretical_eigvals(inputs, labels, model, loss_type, device):
    X_spec_norm = torch.linalg.matrix_norm(inputs, 2)
    all_layers = model._all_layers()
    L, W = len(all_layers), all_layers[0]
    W_spec_norm = torch.linalg.matrix_norm(W.weight.data, 2)
    return (X_spec_norm**2 / inputs.size(0) * L * W_spec_norm**(2*(L-1))).cpu().numpy()


if __name__ == "__main__":

    source_name = os.path.splitext(os.path.basename(__file__))[0]

    DATASETS = (datasets.MNIST, datasets.FashionMNIST, datasets.CIFAR10)
    OUTPUT_DIMS = (10, 10, 10)
    MAX_PROP = None
    WIDTHS = NARROW_DEEP_WIDTHS
    LR = NARROW_DEEP_LR

    fig, axs = plt.subplots(1, len(DATASETS), figsize=(AX_WIDTH*len(DATASETS), AX_HEIGHT*1))
    if not hasattr(axs, "__len__"):
        axs = np.array((axs,))

    for Dataset, output_dim, ax in zip(DATASETS, OUTPUT_DIMS, axs):

        kwargs = dict(
            Dataset=Dataset, output_dim=output_dim, subset_size=SUBSET_SIZE,
            sigma_x=SIGMA_X, max_prop=MAX_PROP,
            Model=MODEL, model_kwargs=MODEL_KWARGS, widths=WIDTHS, balanced_init=False,
            act_name=ACTIVATION, act_kwargs=ACT_KWARGS,
            loss_type=MSE_LOSS, lr=LR, sgd_kwargs=SGD_KWARGS,
            epochs=EPOCHS, log_every=LOG_EVERY, device=DEVICE, seed=SEED,
            tracker_fns=(
                partial(compute_numerical_eigvals, n_eigs=1),
                compute_theoretical_eigvals,
            )
        )
        assets, out = train_gd(**kwargs)
        fn = f"{RESULTS}/{source_name}/{Dataset.__name__.lower()}.pt"
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        torch.save(dict(kwargs=kwargs, assets=assets, out=out), fn)

        *_, numerical_eigvals, theoretical_eigvals = out
        ax.plot(
            LOG_EVERY*np.arange(0, len(numerical_eigvals)),
            numerical_eigvals,
            linewidth=LINE_WIDTH, color="black", label="Sharpness"
        )
        ax.plot(
            LOG_EVERY*np.arange(0, len(theoretical_eigvals)),
            theoretical_eigvals[:,0],
            linewidth=LINE_WIDTH, linestyle="--", color="blue",
            label=r"$\|W_1\|_2^2+\|W_2\|_2^2$"
        )
        ax.plot(
            LOG_EVERY*np.arange(0, len(theoretical_eigvals)),
            theoretical_eigvals[:,1],
            linewidth=LINE_WIDTH, linestyle="--", color="brown",
            label=r"$\max\{\|W_1\|_2^2, \|W_2\|_2^2\}$"
        )
        ax.tick_params(axis="both", labelsize=LABEL_SIZE)
        ax.set_xlabel("Epoch", fontsize=FONT_SIZE)
        ax.grid()

    axs[0].legend(loc="upper right", fontsize=FONT_SIZE, framealpha=FRAME_ALPHA)

    fig.tight_layout()
    fn = f"{ASSETS}/{source_name}.png"
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    plt.savefig(fn)
    plt.close()
    print(f"\nPlot saved at {fn}")