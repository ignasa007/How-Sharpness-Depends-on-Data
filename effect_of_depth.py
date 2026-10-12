from os.path import basename, splitext

import numpy as np
import matplotlib.pyplot as plt

from constants import *


ratios = np.array([0.2, 0.5, 5., 10.])
Ls = np.array([1, 2, 4, 8, 16, 32, 64])


fig, ax = plt.subplots(1, 1, figsize=(7.2*1, 4.8*1))
for ratio in ratios:
    ax.plot(
        Ls, Ls * ratio**(2/Ls) * 0.2**(1-1/Ls),
        linewidth=5, marker="o", markersize=8, label=rf"$\|X\|^2_2/n = {ratio}$"
    )

ax.tick_params(axis="both", labelsize=15)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel(r"Depth", size=18)
ax.set_ylabel(r"Sharpness", size=18)
ax.grid()
ax.legend(fontsize=18, framealpha=1.)

fig.tight_layout()
plt.savefig(f"{ASSETS}/{splitext(basename(__file__))[0]}.png")
plt.close()