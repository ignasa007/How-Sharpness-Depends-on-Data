import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from constants import *
from utilities import pad


n, L = 100, 5
d_in, d_hid, d_out = 20, 15, 10
sigma_x = 3

assert n >= d_in
widths = [d_in] + [d_hid]*(L-1) + [d_out]
k = min(widths)
X_spec_norm = n**0.5 * sigma_x

# 1. Generate whitened X
Q_x, _ = torch.linalg.qr(torch.randn(n, d_in), mode="reduced")
X = Q_x.T * X_spec_norm
print(f"Deviation from Assumption 1: {(
    X@X.T / X_spec_norm**2 - torch.eye(X.size(0))
).abs().max().item():.3e}")

# 2. Fixed Singular Values for all layers
s_sq = torch.sort(torch.rand(k), descending=True).values
s = torch.sqrt(s_sq)

# 3. Generate Aligned Weights: V_{l+1} = U_l
weights = []
V, _ = torch.linalg.qr(torch.randn(widths[0], k))
for l in range(L):
    U, _ = torch.linalg.qr(torch.randn(widths[l+1], k))
    V, W = U, (U*s) @ V.T
    W = W.clone().requires_grad_(True)
    weights.append(W)
with torch.no_grad():
    for l, (W1, W2) in enumerate(zip(weights[:-1], weights[1:]), 1):
        print(f"Imbalance between layer {l} and layer {l+1}: {(
            W1 @ W1.T - W2.T @ W2
        ).abs().max().item():.3e}")

# 4. Target Generation (Zero Loss)
weights_prod = torch.eye(weights[0].size(1))
with torch.no_grad():
    for W in weights:
        weights_prod = W @ weights_prod
    Y = weights_prod @ X

print(f"Deviation from OLS solution: {(
    Y@X.T / X_spec_norm**2 - weights_prod
).abs().max().item():.3e}")
print(f"Error in Equation 20b: {(
    pad(torch.pow(s_sq, len(weights)), d_out) -
    pad(torch.linalg.svdvals(Y).square(), d_out) / X_spec_norm**2
).abs().max().item():.3e}")

# 5. Loss and Autograd Hessian
def get_loss(flat_params):
    out, pointer = X, 0
    for l in range(len(weights)):
        W = flat_params[pointer:pointer+weights[l].numel()].view_as(weights[l])
        out = W @ out
        pointer += weights[l].numel()
    return 0.5 / n * torch.sum((out-Y)**2)

flat_W = torch.cat([W.flatten() for W in weights])
H = torch.autograd.functional.hessian(get_loss, flat_W)
eigs_actual = torch.linalg.eigvalsh(H)

# 6. Geometric Series
analytical = []
for i in range(d_in):
    for j in range(d_out):
        a, b = s_sq[i] if i < len(s_sq) else 0., s_sq[j] if j < len(s_sq) else 0.
        val = (a**L-b**L)/(a-b) if abs(a-b) > 1e-7 else L*(a**(L-1))
        analytical.append(sigma_x**2*val)

eigs_theory = np.sort(analytical, descending=True)
eigs_actual = eigs_actual.flip(0).numpy()

k = 10
print(f"\nAlignment Verification (L = {len(weights)}, widths = {", ".join(map(str, widths))})")
print(f"Displaying Top-{k} and Bottom-{k} Eigvals")
print("-" * 75)
print(f"Numerical Eigvals:\n{eigs_actual[:k]} ...\n{eigs_actual[d_in*d_out-k:d_in*d_out]}")
print("-" * 75)
print(f"Analytical Eigvals:\n{eigs_theory[:k]} ...\n{eigs_theory[-k:]}")

fig, ax = plt.subplots(1, 1, figsize=(AX_WIDTH*1, AX_HEIGHT*1))
ax.plot(range(1, len(eigs_actual)+1), eigs_actual, linewidth=7.5, label="Computed Eigenvalues")
ax.plot(range(1, len(eigs_theory)+1), eigs_theory, linewidth=5, linestyle="--", label="Theoretical Predictions")
ax.tick_params(axis="both", labelsize=15)
ax.set_xlabel(r"Index, $i$", fontsize=18)
ax.set_ylabel(r"Eigenvalue, $\lambda_i$", fontsize=18)
ax.set_xscale("log")
ax.grid()
ax.legend(loc="upper right", framealpha=1., fontsize=18)

fig.tight_layout()
fn = f"{ASSETS}/{os.path.splitext(os.path.basename(__file__))[0]}.png"
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn)
plt.close()
print(f"\nPlot saved at {fn}")