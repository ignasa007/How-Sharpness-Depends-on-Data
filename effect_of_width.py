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


source_name = os.path.splitext(os.path.basename(__file__))[0]

def main(Dataset, feat_transform, output_dim, L, d, loss_type):

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
    ols_loss = loss_fn(x@w, y, loss_type=loss_type).item()
    print(f"OLS regression loss: {ols_loss:.6e}")

    u_1, s_W, v_L = torch.linalg.svd(w, full_matrices=False)    # (d_0, d_L), (d_L,), (d_L, d_L), assuming d_L < d_0
    s_W = s_W ** (1/L)  # Note: don't diag-embed here because torch computes 0^0 = 1
    v_L = v_L.T         # torch.linalg.svd returns the transposed right SVs
    
    # Bounds
    feat_corr = 1/x.size(0) * x.T @ x
    lower_bound = torch.lobpcg(feat_corr)[0].item() * s_W.max().item()**(2*(L-1))
    upper_bound = L * lower_bound

    # Hessian sharpness
    weights = list()
    out_dims = [d]*(L-1) + [output_dim]
    k = min(out_dims)
    u_1, s_W, v_L = u_1[:, :k], s_W[:k], v_L[:, :k]
    u = u_1.clone()
    for out_dim in out_dims:
        v, _ = torch.linalg.qr(torch.randn(out_dim, k, device=DEVICE))
        w = (u*s_W) @ v.T
        weights.append(w)
        u = v
    weights[-1] = w @ v @ v_L.T
    model = list()
    with torch.no_grad():
        for w in weights:
            layer = nn.Linear(*w.size(), bias=False)
            layer.weight.copy_(w.T)
            model.append(layer)
    model = nn.Sequential(*model)
    model = model.to(DEVICE)
    with torch.no_grad():
        simulated_loss = loss_fn(model(x), y, loss_type=loss_type).item()
        print(f"Simulated OLS solution's loss: {simulated_loss:.6e}")
    p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hvp_delta = lambda vector: compute_hvp(x, y, model, loss_type, vector, DEVICE).detach().cpu()
    hessian_sharpness, _ = lanczos(tensor_matvec=hvp_delta, p=p, n_eigs=1)
    hessian_sharpness = hessian_sharpness.item()
    
    print(f"Lower bound: {lower_bound:.6e}")
    print(f"Hessian sharpness: {hessian_sharpness:.6e}")
    print(f"Upper bound: {upper_bound:.6e}")

    return ols_loss, simulated_loss, lower_bound, hessian_sharpness, upper_bound

def identity(x):
    return x

def whiten(x):
    u, _, vt = torch.linalg.svd(x, full_matrices=False)
    x = (u * np.sqrt(x.size(0))) @ vt
    return x

Dataset, output_dim = datasets.MNIST, 10
TRANSFORM = identity
DEPTHS = (2, 6)
WIDTHS = (2, 4, 8, 16, 32, 64)
LOSS_TYPE = MSE_LOSS

outs = list()
for depth in DEPTHS:
    for width in WIDTHS:
        outs.append(main(Dataset, TRANSFORM, output_dim, depth, width, LOSS_TYPE))
        print()
ols_losses, simulated_losses, lower_bounds, hessian_sharpnesses, upper_bounds = zip(*outs)

fig, axs = plt.subplots(1, 2, figsize=(AX_WIDTH*2, AX_HEIGHT*1))
axs[0].vlines(
    output_dim, min(simulated_losses), max(simulated_losses),
    color="gray", linewidth=LINE_WIDTH, linestyle="--"
)
axs[0].hlines(
    ols_losses[0], WIDTHS[0], WIDTHS[-1],
    color="black", linewidth=LINE_WIDTH, linestyle="--", zorder=2.05, label="OLS"
)
axs[0].plot(
    WIDTHS, simulated_losses[:len(WIDTHS)],
    marker="o", markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label="Simulated"
)
axs[0].set_ylabel("Mean Squared Error", fontsize=FONT_SIZE)
for i, depth in enumerate(DEPTHS):
    axs[1].plot(
        WIDTHS, hessian_sharpnesses[len(WIDTHS)*i:len(WIDTHS)*(i+1)],
        marker="o", markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=f"$L = {depth}$"
    )
axs[1].set_ylabel("Hessian Sharpness", fontsize=FONT_SIZE)

for ax in axs:
    ax.set_xticks(WIDTHS)
    ax.set_xscale("log", base=2)
    ax.tick_params(axis="both", labelsize=LABEL_SIZE)
    ax.set_xlabel("Width", fontsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE, framealpha=FRAME_ALPHA)
    # ax.grid()

fig.tight_layout()
fn = f"{ASSETS}/{source_name}.png"
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn)
plt.close()
print(f"Plot saved at {fn}")