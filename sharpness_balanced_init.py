import os

import numpy as np
from torch.nn.utils import parameters_to_vector
from torchvision import datasets
import matplotlib.pyplot as plt

from utilities import pad, compute_hvp, lanczos, train_gd
from constants import *


def compute_numerical_eigvals(inputs, labels, model, loss_type, device, n_eigs=1):
    model.zero_grad()
    parameters = parameters_to_vector(model.parameters()).detach().cpu()
    hvp_delta = lambda vector: compute_hvp(inputs, labels, model, loss_type, vector, device).detach().cpu()
    eigenvalues, _ = lanczos(tensor_matvec=hvp_delta, p=len(parameters), n_eigs=n_eigs)
    return eigenvalues.numpy()

@torch.no_grad()
def compute_theoretical_eigvals(inputs, labels, model, loss_type, device, n_eigs=1):
    L = len(list(model.parameters()))
    X_spec_norm = torch.linalg.matrix_norm(inputs, 2)
    sigma_2L = (torch.linalg.svdvals(labels) / X_spec_norm).square()
    sigma_2 = sigma_2L ** (1/L)
    num = pad(sigma_2L, inputs.size(1)).unsqueeze(0) - pad(sigma_2L, labels.size(1)).unsqueeze(1)
    den = pad(sigma_2, inputs.size(1)).unsqueeze(0) - pad(sigma_2, labels.size(1)).unsqueeze(1)
    result = X_spec_norm.square() / inputs.size(0) \
        * torch.where(den.abs()>1e-8, num/den, (L*sigma_2L/sigma_2).unsqueeze(1))
    return torch.sort(result, descending=True).values.to("cpu").flatten()[:n_eigs].numpy()


if __name__ == "__main__":

    source_name = os.path.splitext(os.path.basename(__file__))[0]

    DATASETS = (datasets.MNIST, datasets.FashionMNIST, datasets.CIFAR10)
    OUTPUT_DIMS = (10, 10, 10)
    MAX_PROPS = np.linspace(0.1, 1.0, 10)
    WIDTHS = NARROW_DEEP_WIDTHS
    LR = NARROW_DEEP_LR

    fig, ax = plt.subplots(1, 1, figsize=(AX_WIDTH*1, AX_HEIGHT*1))

    for Dataset, output_dim in zip(DATASETS, OUTPUT_DIMS):
        numerical_eigvals, theoretical_eigvals = list(), list()
        for max_prop in MAX_PROPS:
            kwargs = dict(
                Dataset=Dataset, output_dim=output_dim, subset_size=SUBSET_SIZE,
                sigma_x=SIGMA_X, max_prop=max_prop,
                Model=MODEL, model_kwargs=MODEL_KWARGS, widths=WIDTHS, balanced_init=True,
                act_name=ACTIVATION, act_kwargs=ACT_KWARGS,
                loss_type=MSE_LOSS, lr=LR, sgd_kwargs=SGD_KWARGS,
                epochs=EPOCHS, log_every=EPOCHS, device=DEVICE, seed=SEED,      # Log at the end of training only
                tracker_fns=(compute_numerical_eigvals, compute_theoretical_eigvals),
                use_ols_soln=True
            )
            assets, out = train_gd(**kwargs)
            fn = f"{RESULTS}/{source_name}/{Dataset.__name__.lower()}/max_prop={max_prop:.1f}.pt"
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            torch.save(dict(kwargs=kwargs, assets=assets, out=out), fn)  
            *_, numerical_eigval, theoretical_eigval = out
            numerical_eigvals.append(numerical_eigval[-1].item())
            theoretical_eigvals.append(theoretical_eigval[-1].item())
        ax.plot(MAX_PROPS, numerical_eigvals, linewidth=LINE_WIDTH, marker="o", ms=MARKER_SIZE, label=Dataset.__name__)
    ax.plot(MAX_PROPS, theoretical_eigvals, linewidth=LINE_WIDTH, marker="o", ms=MARKER_SIZE, color="black", label="Predictions")

    ax.set_xticks(MAX_PROPS[1::2])
    ax.tick_params(axis="both", labelsize=LABEL_SIZE)
    ax.set_xlabel("Maximum Label Proportion", fontsize=FONT_SIZE)
    ax.set_ylabel("Final Sharpness", fontsize=FONT_SIZE)
    ax.legend(loc="upper left", fontsize=FONT_SIZE, framealpha=FRAME_ALPHA)
    ax.grid()

    fig.tight_layout()
    fn = f"{ASSETS}/{source_name}.png"
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    plt.savefig(fn)
    plt.close()
    print(f"\nPlot saved at {fn}")