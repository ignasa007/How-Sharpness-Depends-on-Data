import os
from inspect import signature

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from utilities import loss_fn, compute_hvp, lanczos
from constants import *


DATASETS = (datasets.MNIST, datasets.FashionMNIST, datasets.CIFAR10)
OUTPUT_DIMS = (10, 10, 10)
d, L = 128, 2
fig, axs = plt.subplots(1, 3, figsize=(AX_WIDTH*3, AX_HEIGHT*1))
width = 0.8
hessian_sharpness = None    # Set to any value to not compute sharpness


def main(Dataset, feat_transform, output_dim, L, d):

    print(f"*** Dataset = {Dataset.__name__}, feat_transform = {feat_transform.__name__}{signature(feat_transform)}, width = {d}, depth = {L} ***")

    full_trainset = Dataset(
        root=f"{DATASTORE}/{Dataset.__name__.lower()}",
        train=True, download=True, transform=transforms.ToTensor()
    )
    data_loader = torch.utils.data.DataLoader(full_trainset, batch_size=SUBSET_SIZE)
    x, y = next(iter(data_loader))

    # Note: we work with transposed coordinates, in comparison with the theory
    x = x.view(x.size(0), -1).float().to(DEVICE)                # (n, d_0)
    x = feat_transform(x)
    y = F.one_hot(y, num_classes=output_dim).float().to(DEVICE) # (n, d_L)
    w = torch.linalg.pinv(x) @ y                                # (d_0, d_L)
    loss = loss_fn(x@w, y, loss_type="Mean Squared Error")
    print(f"OLS regression loss: {loss.item():.6e}")

    u_1, s_W, v_L = torch.linalg.svd(w, full_matrices=False)    # (d_0, d_L), (d_L,), (d_L, d_L), assuming d_L < d_0
    s_W = s_W ** (1/L)  # Note: don't diag-embed here because torch computes 0^0 = 1
    v_L = v_L.T         # torch.linalg.svd returns the transposed right SVs
    print(f"Distance of u_1 from orthogonality: {torch.max(torch.abs(u_1.T@u_1-torch.eye(output_dim, device=DEVICE))).item():.6e}")

    # Bounds
    feat_corr = 1/x.size(0) * x.T @ x
    lower_bound = torch.lobpcg(feat_corr)[0].item() * s_W.max().item()**(2*(L-1))
    upper_bound = L * lower_bound

    # NTK sharpness
    ntk, xu_1 = 0., x @ u_1
    for l in range(1, L+1):
        left_gram = 1/x.size(0) * (x @ x.T if l == 1 else (xu_1 * s_W**(2*(l-1))) @ xu_1.T)
        right_gram = (v_L * s_W**(2*(L-l))) @ v_L.T
        ntk += torch.kron(left_gram, right_gram)
    ntk_sharpness = torch.lobpcg(ntk)[0].item()

    # Hessian sharpness
    if "hessian_sharpness" in globals():
        hessian_sharpness = globals()["hessian_sharpness"]
    else:
        assert d >= output_dim
        weights = list()
        out_dims = [d]*(L-1) + [output_dim]
        u = u_1.clone()
        for ell, out_dim in enumerate(out_dims, 1):
            v, _ = torch.linalg.qr(torch.randn(out_dim, output_dim, device=DEVICE))
            print(f"Distance of v_{ell} from orthogonality: {torch.max(torch.abs(v.T@v-torch.eye(output_dim, device=DEVICE))).item():.6e}")
            w = (u*s_W) @ v.T
            weights.append(w)
            u = v
        weights[-1] = w @ v @ v_L.T
        for ell, (w_1, w_2) in enumerate(zip(weights[:-1], weights[1:]), 1):
            print(f"Imbalance between layer {ell} and {ell+1}: {torch.max(torch.abs(w_1.T@w_1-w_2@w_2.T)).item():.6e}")
        model = list()
        with torch.no_grad():
            for w in weights:
                layer = nn.Linear(*w.size(), bias=False)
                layer.weight.copy_(w.T)
                model.append(layer)
        model = nn.Sequential(*model)
        model = model.to(DEVICE)
        with torch.no_grad():
            loss = loss_fn(model(x), y, loss_type="Mean Squared Error")
            print(f"Simulated OLS solution's loss: {loss.item():.6e}")
        p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        hvp_delta = lambda vector: compute_hvp(x, y, model, vector).detach().cpu()
        hessian_sharpness, _ = lanczos(tensor_matvec=hvp_delta, p=p, n_eigs=1)
        hessian_sharpness = hessian_sharpness.item()
        del model

    return lower_bound, ntk_sharpness, hessian_sharpness, upper_bound

def identity(x):
    return x

def whiten(x):
    u, _, vt = torch.linalg.svd(x, full_matrices=False)
    x = (u * np.sqrt(x.size(0))) @ vt
    return x

outs = list()
for Dataset, output_dim in zip(DATASETS, OUTPUT_DIMS):
    outs.append(main(Dataset, identity, output_dim, L, d))
    print()

lower_bounds, ntk_sharpnesses, hessian_sharpnesses, upper_bounds = map(np.array, zip(*outs))
objs = (
    (lower_bounds, "Lower Bound"),
    (ntk_sharpnesses, "NTK Sharpness"),
    # (hessian_sharpnesses, "Hessian Sharpness"),   # Plot to verify that the NTK sharpness has been correctly computed
    (upper_bounds, "Upper Bound"),
)

ax = axs[0]
for i, (obj, label) in enumerate(objs):
    ax.bar(np.arange(len(DATASETS))+(-len(objs)+1+2*i)*width/(2*len(objs)), obj, width=width/len(objs), label=label)

ax.set_xticks(ticks=np.arange(len(DATASETS)), labels=list(map(lambda x: x.__name__, DATASETS)), fontsize=18)
ax.tick_params(axis="y", labelsize=15)
ax.set_yscale("log")
ax.legend(loc="upper left", fontsize=18, framealpha=1.0)

outs = list()
for Dataset, output_dim in zip(DATASETS, OUTPUT_DIMS):
    outs.append(main(Dataset, whiten, output_dim, L, d))
    print()

lower_bounds, ntk_sharpnesses, hessian_sharpnesses, upper_bounds = map(np.array, zip(*outs))
objs = (
    (lower_bounds, "Lower Bound"),
    (ntk_sharpnesses, "NTK Sharpness"),
    (upper_bounds, "Upper Bound"),
)

ax = axs[1]
for i, (obj, label) in enumerate(objs):
    ax.bar(np.arange(len(DATASETS))+(-len(objs)+1+2*i)*width/(2*len(objs)), obj, width=width/len(objs), label=label)
ax.set_xticks(ticks=np.arange(len(DATASETS)), labels=list(map(lambda x: x.__name__, DATASETS)), fontsize=18)
ax.tick_params(axis="y", labelsize=15)
ax.set_yscale("log")

Dataset, output_dim = datasets.MNIST, 10
pert_rads = np.linspace(0., 0.35, 8)

outs = list()
for pert_rad in pert_rads:
    def random_spectrum(x, pert_rad=round(float(pert_rad), 3)):
        u, s, vt = torch.linalg.svd(x, full_matrices=False)
        torch.manual_seed(0)
        x = (u * np.sqrt(x.size(0)) * torch.clip(2.5+pert_rad*torch.randn_like(s), min=0.)) @ vt
        return x
    outs.append(main(Dataset, random_spectrum, output_dim, L, d))
    print()

lower_bounds, ntk_sharpnesses, hessian_sharpnesses, upper_bounds = map(np.array, zip(*outs))
objs = (
    (lower_bounds, "Lower Bound"),
    (ntk_sharpnesses, "NTK Sharpness"),
    (upper_bounds, "Upper Bound"),
)

ax = axs[2]
for i, (obj, label) in enumerate(objs):
    ax.plot(pert_rads, obj, linewidth=5, marker="o", markersize=8, label=label)

ax.tick_params(axis="both", labelsize=15)
ax.set_yscale("log")
ax.set_xlabel("Std. Dev. for SV Sampling", fontsize=18)

fig.tight_layout()
fn = f"{ASSETS}/{os.path.splitext(os.path.basename(__file__))[0]}.png"
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn)
plt.close()
print(f"\nPlot saved at {fn}")