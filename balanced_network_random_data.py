from os.path import basename, splitext

import numpy as np
import torch
import matplotlib.pyplot as plt

from constants import *


n, d, L = 100, 20, 5
sigma_x = 3


# 1. Generate whitened X
X_raw = torch.randn(d, n)
Q_x, _ = torch.linalg.qr(X_raw.T)
X = Q_x.T[:d, :] * np.sqrt(n) * sigma_x
print((1/n*X@X.T - sigma_x**2*torch.eye(d)).abs().max())

# 2. Fixed Singular Values for all layers
s_sq = torch.sort(torch.rand(d), descending=True).values
S = torch.diag(torch.sqrt(s_sq))

# 3. Generate Aligned Weights: V_{l+1} = U_l
weights = []
V, _ = torch.linalg.qr(torch.randn(d, d))
for l in range(L):
    U, _ = torch.linalg.qr(torch.randn(d, d))
    V, W = U, U@S@ V.T
    W = W.clone().requires_grad_(True)
    weights.append(W)
with torch.no_grad():
    for W1, W2 in zip(weights[:-1], weights[1:]):
        print((W1@W1.T-W2.T@W2).abs().max())

# 4. Target Generation (Zero Loss)
weights_prod = torch.eye(d)
with torch.no_grad():
    for W in weights:
        weights_prod = W @ weights_prod
Y = weights_prod @ X

print((1/n*Y@X.T - sigma_x**2*weights_prod).abs().max())
print((torch.pow(s_sq, L) - torch.linalg.svdvals(Y).square() / (n*sigma_x**2)).abs().max())

# 5. Loss and Autograd Hessian
def get_loss(flat_params):
    curr_X = X
    curr = 0
    for _ in range(L):
        W = flat_params[curr:curr+d*d].view(d, d)
        curr_X = W @ curr_X
        curr += d*d
    return 0.5 / n * torch.sum((curr_X-Y)**2)

flat_W = torch.cat([W.flatten() for W in weights])
H = torch.autograd.functional.hessian(get_loss, flat_W)
eigs_actual = torch.linalg.eigvalsh(H)

# 6. Geometric Series
analytical = []
for i in range(d):
    for j in range(d):
        a, b = s_sq[i], s_sq[j]
        val = (a**L - b**L)/(a - b) if abs(a-b) > 1e-7 else L*(a**(L-1))
        analytical.append(sigma_x**2*val)

eigs_theory = torch.tensor(sorted(analytical, reverse=True))
eigs_actual_top = torch.sort(eigs_actual, descending=True).values

k = 10
print(f"Alignment Verification (L={L}, d={d})\nDisplaying Top-{k} and Bottom-{k} Eigvals")
print("\n" + "-" * 50)
print("\nNumerical Eigvals:\n")
print(eigs_actual_top[:k].numpy(), " ...\n\n", eigs_actual_top[len(eigs_theory)-k:len(eigs_theory)].numpy(), sep="")
print("\nAnalytical Eigvals:\n")
print(eigs_theory[:k].numpy(), " ...\n\n", eigs_theory[-k:].numpy(), sep="")

s_sq = (torch.linalg.svdvals(Y).square() / (n*sigma_x**2)) ** (1/L)
analytical = []
for i in range(d):
    for j in range(d):
        a, b = s_sq[i], s_sq[j]
        val = (a**L - b**L)/(a - b) if abs(a-b) > 1e-7 else L*(a**(L-1))
        analytical.append(sigma_x**2*val)
eigs_theory = torch.tensor(sorted(analytical, reverse=True))
print("Analytical Eigvals:\n")
print(eigs_theory[:k].numpy(), " ...\n\n", eigs_theory[-k:].numpy(), sep="")

fig, ax = plt.subplots(1, 1, figsize=(AX_WIDTH*1, AX_HEIGHT*1))
ax.plot(range(1, len(eigs_actual_top)+1), eigs_actual_top, linewidth=7.5, label="Computed Eigenvalues")
ax.plot(range(1, len(eigs_theory)+1), eigs_theory, linewidth=5, linestyle="--", label="Theoretical Predictions")
ax.tick_params(axis="both", labelsize=15)
ax.set_xlabel(r"Index, $i$", fontsize=18)
ax.set_ylabel(r"Eigenvalue, $\lambda_i$", fontsize=18)
ax.set_xscale("log")
ax.grid()
ax.legend(loc="upper right", framealpha=1., fontsize=18)

fig.tight_layout()
plt.savefig(f"{ASSETS}/{splitext(basename(__file__))[0]}.png")
plt.close()