import os

import numpy as np
from torch.nn.utils import parameters_to_vector
from torchvision import datasets
import matplotlib.pyplot as plt

from utilities import compute_hvp, lanczos, train_gd
from constants import *


DATASETS = (datasets.MNIST, datasets.FashionMNIST, datasets.CIFAR10)
OUTPUT_DIMS = (10, 10, 10)
MAX_PROPS = np.linspace(0.1, 1.0, 10)
WIDTHS = NARROW_DEEP_WIDTHS
LR = NARROW_DEEP_LR


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

fig, ax = plt.subplots(1, 1, figsize=(AX_WIDTH*1, AX_HEIGHT*1))

for Dataset, output_dim in zip(DATASETS, OUTPUT_DIMS):
    numerical_eigvals, theoretical_eigvals = list(), list()
    for max_prop in MAX_PROPS:
        _, out = train_gd(
            Dataset=Dataset, output_dim=output_dim, subset_size=SUBSET_SIZE,
            sigma_x=SIGMA_X, max_prop=max_prop,
            Model=MODEL, model_kwargs=MODEL_KWARGS, widths=WIDTHS, balanced_init=True,
            act_name=ACTIVATION, act_kwargs=ACT_KWARGS,
            loss_type=LOSS_TYPE, lr=LR, sgd_kwargs=SGD_KWARGS,
            epochs=EPOCHS, log_every=EPOCHS, device=DEVICE, seed=SEED,      # Log at the end of training only
            tracker_fns=(compute_numerical_eigvals, compute_theoretical_eigvals)
        )
        *_, numerical_eigval, theoretical_eigval = out
        numerical_eigvals.append(numerical_eigval[-1].item())
        theoretical_eigvals.append(theoretical_eigval[-1].item())
    ax.plot(MAX_PROPS, numerical_eigvals, linewidth=5, marker="o", ms=8, mew=3, alpha=1.0, label=Dataset.__name__)
ax.plot(MAX_PROPS, theoretical_eigvals, linewidth=5, marker="o", ms=8, mew=3, color="black", alpha=1.0, label="Predictions")

ax.set_xticks(MAX_PROPS)
ax.tick_params(axis="both", labelsize=15)
ax.set_xlabel("Maximum Label Proportion", fontsize=18)
ax.set_ylabel("Final Sharpness", fontsize=18)
ax.legend(loc="upper left", fontsize=18, framealpha=1.0)
ax.grid()

fig.tight_layout()
fn = f"{ASSETS}/{os.path.splitext(os.path.basename(__file__))[0]}-LR={LR}.png"
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn)
plt.close()
print(f"\nPlot saved at {fn}")