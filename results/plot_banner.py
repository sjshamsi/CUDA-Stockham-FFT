import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Conceptual banner: our test signal -> Stockham radix-2 stages -> spectrum.
# The spectrum is computed with numpy at a small N, not taken from the GPU runs.
here = Path(__file__).parent
N = 1024
t = np.arange(N) / N
x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
X = np.abs(np.fft.fft(x))

fig = plt.figure(figsize=(13, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.25, 1.1], wspace=0.25)
ax_sig, ax_fly, ax_spec = (fig.add_subplot(gs[i]) for i in range(3))

# Left: the input signal
ax_sig.plot(t, x, color="#3b6ea5", lw=2)
ax_sig.set_title(r"input: $\sin(2\pi\,5t) + 0.5\,\sin(2\pi\,10t)$", fontsize=11)
ax_sig.set_xlabel("t")
ax_sig.set_yticks([])
for s in ("top", "right", "left"):
    ax_sig.spines[s].set_visible(False)

# Middle: data flow of the Stockham kernel for n = 8 (3 stages, ping-pong buffers)
n = 8
stages = int(np.log2(n))
colors = ["#3b6ea5", "#e0a33b", "#d62728"]
for s in range(stages):
    p = 2 ** s
    for th in range(n // 2):          # one thread per butterfly
        j = th % p
        src = (th, th + n // 2)       # reads: contiguous halves
        dst = (2 * th - j, 2 * th - j + p)  # writes: autosorted positions
        for a in src:
            for b in dst:
                ax_fly.add_patch(FancyArrowPatch((s, n - 1 - a), (s + 1, n - 1 - b), arrowstyle="-",
                                                 color=colors[s], lw=1.1, alpha=0.75))
for c in range(stages + 1):
    ax_fly.scatter([c] * n, range(n), s=28, color="#222222", zorder=3)
for s in range(stages):
    ax_fly.text(s + 0.5, n - 0.35, f"stage {s + 1}\np = {2 ** s}", ha="center", fontsize=8.5, color=colors[s])
ax_fly.set_xlim(-0.2, stages + 0.2)
ax_fly.set_ylim(-0.6, n + 0.6)
ax_fly.set_title("Stockham radix-2 butterflies (n = 8)", fontsize=11)
ax_fly.axis("off")

# Right: the spectrum
k = np.arange(31)
ax_spec.vlines(k, 0, X[k], color="#d62728", lw=2)
ax_spec.plot(k, X[k], "o", color="#d62728", ms=4)
for kk in (5, 10):
    ax_spec.annotate(f"k = {kk}", (kk, X[kk]), textcoords="offset points", xytext=(6, -2), fontsize=9)
ax_spec.set_title(r"output: $|X_k|$", fontsize=11)
ax_spec.set_xlabel("frequency bin k")
ax_spec.set_yticks([])
for s in ("top", "right", "left"):
    ax_spec.spines[s].set_visible(False)

# Arrows between panels
for x0, x1 in ((0.345, 0.375), (0.64, 0.67)):
    fig.patches.append(FancyArrowPatch((x0, 0.5), (x1, 0.5), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=18, color="#555555", lw=1.5))

fig.savefig(here / "banner.png", dpi=150, bbox_inches="tight")
print("wrote", here / "banner.png")
