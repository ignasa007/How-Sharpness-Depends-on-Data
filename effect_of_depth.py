import os

import numpy as np
import matplotlib.pyplot as plt

from constants import *


source_name = os.path.splitext(os.path.basename(__file__))[0]

ratios = np.array([0.5, 5., 10.])
Ls = np.array([1, 2, 4, 8, 16, 32, 64])


fig, ax = plt.subplots(1, 1, figsize=(AX_WIDTH*1, AX_HEIGHT*1))
for ratio in ratios:
    ax.plot(
        Ls, Ls * ratio**(2/Ls) * 0.2**(1-1/Ls),
        linewidth=LINE_WIDTH, marker="o", markersize=MARKER_SIZE, label=rf"$\|X\|^2_2/n = {ratio}$"
    )

ax.tick_params(axis="both", labelsize=LABEL_SIZE)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Depth", fontsize=FONT_SIZE)
ax.set_ylabel("Predicted Sharpness", fontsize=FONT_SIZE)
# ax.grid()
ax.legend(loc="lower right", fontsize=FONT_SIZE, framealpha=FRAME_ALPHA)

fig.tight_layout()
fn = f"{ASSETS}/{source_name}.png"
os.makedirs(os.path.dirname(fn), exist_ok=True)
plt.savefig(fn)
plt.close()
print(f"\nPlot saved at {fn}")