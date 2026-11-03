import os

import numpy as np
import torch
from torchvision import datasets
import matplotlib.pyplot as plt

from utilities import sq_parameter_norm, compute_hvp, train_gd
from constants import *


def compute_numerical_eigvals(inputs, labels, model, loss_type, device):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sum_samples = 0.
    for _ in range(1, HUTCHINSON_SAMPLES+1):
        random_vector = torch.randn(num_params, device=device)
        sum_samples += random_vector @ compute_hvp(inputs, labels, model, random_vector, device)
    sum_eigenvalues = sum_samples / HUTCHINSON_SAMPLES
    return sum_eigenvalues.item()


if __name__ == "__main__":

    source_name = os.path.splitext(os.path.basename(__file__))[0]

    Dataset = datasets.CIFAR10
    OUTPUT_DIM = 10
    MAX_PROP = None
    WIDTHS = (WIDE_SHALLOW_WIDTHS, NARROW_DEEP_WIDTHS)
    SAVE_NAMES = ("wide-shallow", "narrow-deep")
    LR = 3e-3

    fig, axs = plt.subplots(1, len(WIDTHS), figsize=(AX_WIDTH*len(WIDTHS), AX_HEIGHT*1))
    if not hasattr(axs, "__len__"):
        axs = np.array((axs,))

    for widths, save_name, ax in zip(WIDTHS, SAVE_NAMES, axs):

        kwargs = dict(
            Dataset=Dataset, output_dim=OUTPUT_DIM, subset_size=SUBSET_SIZE,
            sigma_x=SIGMA_X, max_prop=MAX_PROP,
            Model=MODEL, model_kwargs=MODEL_KWARGS, widths=widths, balanced_init=False,
            act_name=ACTIVATION, act_kwargs=ACT_KWARGS,
            loss_type=MSE_LOSS, lr=LR, sgd_kwargs=SGD_KWARGS,
            epochs=EPOCHS, log_every=LOG_EVERY, device=DEVICE, seed=SEED,
            tracker_fns=(sq_parameter_norm, compute_numerical_eigvals),
        )
        assets, out = train_gd(**kwargs)
        fn = f"{RESULTS}/{source_name}/{save_name}.pt"
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        torch.save(dict(kwargs=kwargs, assets=assets, out=out), fn)

        *_, sq_param_norms, numerical_eigvals = out
        p = ax.plot(
            LOG_EVERY*np.arange(0, len(numerical_eigvals)),
            numerical_eigvals,
            linestyle="-", linewidth=5, alpha=1.0, color="blue"
        )
        ax.tick_params(axis="both", labelsize=LABEL_SIZE)
        ax.set_xlabel("Epoch", fontsize=FONT_SIZE)
        ax.set_ylabel("Sharpness", fontsize=FONT_SIZE, color=p[-1].get_color())
        ax.grid()
        ax_twin = ax.twinx()
        p = ax_twin.plot(
            LOG_EVERY*np.arange(0, len(sq_param_norms)),
            sq_param_norms.sum(axis=1),
            linestyle="--", linewidth=5, alpha=1.0, color="brown"
        )
        ax_twin.tick_params(axis="both", labelsize=LABEL_SIZE)
        ax_twin.set_ylabel("Parameter Norm", fontsize=FONT_SIZE, color=p[-1].get_color(), rotation=270, labelpad=20)

    fig.tight_layout()
    fn = f"{ASSETS}/{source_name}.png"
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    plt.savefig(fn)
    plt.close()
    print(f"\nPlot saved at {fn}")