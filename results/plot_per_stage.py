import csv
from pathlib import Path
import matplotlib.pyplot as plt

here = Path(__file__).parent
rows = list(csv.DictReader((here / "kernel_launches.csv").open()))
BYTES = 2 * 2**26 * 16  # each stage reads and writes the whole array once
PEAK = 898  # GB/s, memoryBandwidth that nsys reports for the Tesla V100-PCIE-16GB
versions = ["v1", "v1.1", "v1.3", "v1.4a"]  # v1.2 has the same stage kernel as v1.1
colors = {"v1": "#1f77b4", "v1.1": "#ff7f0e", "v1.3": "#d62728", "v1.4a": "#9467bd"}

# cuFFT does the whole transform in 3 kernels, so average its bandwidth over 3 passes
cufft = [r for r in rows if r["version"] == "v1.4a" and r["kernel"] == "regular_fft"]
cufft_bw = 3 * BYTES / (sum(float(r["duration_ms"]) for r in cufft) * 1e-3) / 1e9

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, block in zip(axes, [32, 256, 1024]):
    for v in versions:
        rs = sorted((r for r in rows if r["version"] == v and r["stage"] and int(r["threads_per_block"]) == block),
                    key=lambda r: int(r["stage"]))
        ax.plot([int(r["stage"]) for r in rs],
                [BYTES / (float(r["duration_ms"]) * 1e-3) / 1e9 for r in rs],
                marker=".", color=colors[v], label=v)
    ax.axhline(PEAK, color="k", lw=1, ls="--", label=f"V100 peak ({PEAK} GB/s)")
    ax.axhline(cufft_bw, color="grey", lw=1.2, ls=":", label=f"cuFFT, average per pass ({cufft_bw:.0f} GB/s)")
    ax.set_title(f"{block} threads per block", fontsize=11)
    ax.set_xlabel("stage s (butterfly span p = 2^(s-1))")
    ax.set_xticks([1, 5, 9, 13, 17, 21, 26])
axes[0].set_ylabel("effective bandwidth of each stage [GB/s]")
axes[0].set_ylim(450, 960)
axes[0].legend(loc="lower right", fontsize=8.5, frameon=False, ncol=2)
fig.suptitle("Per-stage bandwidth, N = 2^26, Tesla V100-PCIE-16GB (Nsight Systems kernel timestamps)", fontsize=11.5)
fig.tight_layout()
fig.savefig(here / "per_stage_bandwidth.png", dpi=150)
print("wrote", here / "per_stage_bandwidth.png")
