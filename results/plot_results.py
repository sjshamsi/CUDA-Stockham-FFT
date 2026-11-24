import csv
from pathlib import Path
import matplotlib.pyplot as plt

here = Path(__file__).parent
rows = list(csv.DictReader((here / "summary.csv").open()))
versions = ["v1", "v1.1", "v1.2", "v1.3", "v1.4a"]
labels = {
    "v1": "v1\nradix-2 +\nbit reversal",
    "v1.1": "v1.1\ndevice-side\ninit/check",
    "v1.2": "v1.2\nstreams +\nevents",
    "v1.3": "v1.3\nshared-mem\ntwiddles",
    "v1.4a": "v1.4a\nStockham\n(no bit rev.)",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})

# Left: best configuration per version, split into butterfly stages and bit reversal
best = {v: min((r for r in rows if r["version"] == v), key=lambda r: float(r["fft_kernel_total_ms"]))
        for v in versions}
x = range(len(versions))
stages = [float(best[v]["stage_kernels_ms"]) for v in versions]
bitrev = [float(best[v]["bit_reverse_ms"]) for v in versions]
ax1.bar(x, stages, color="#3b6ea5", label="26 radix-2 stage kernels")
ax1.bar(x, bitrev, bottom=stages, color="#e0a33b", label="bit-reversal kernel")
cufft = float(best["v1.4a"]["cufft_kernel_ms"])
ax1.axhline(cufft, color="#444444", ls="--", lw=1)
ax1.text(len(versions) - 0.5, cufft + 2, f"cuFFT {cufft:.1f} ms", ha="right", va="bottom", fontsize=9)
for i, v in enumerate(versions):
    tot = float(best[v]["fft_kernel_total_ms"])
    ax1.text(i, tot + 1.5, f"{tot:.1f}", ha="center", fontsize=9)
ax1.set_xticks(list(x), [labels[v] for v in versions], fontsize=8.5)
ax1.set_ylabel("GPU kernel time per FFT [ms]")
ax1.set_title("Kernel time per version (best block size)", fontsize=11)
ax1.set_ylim(0, 128)
ax1.legend(loc="upper center", ncol=2, fontsize=8.5, frameon=False)

# Right: sensitivity to threads per block
for v, marker in zip(versions, "osD^v"):
    rs = sorted((r for r in rows if r["version"] == v), key=lambda r: int(r["threads_per_block"]))
    ax2.plot([int(r["threads_per_block"]) for r in rs], [float(r["fft_kernel_total_ms"]) for r in rs],
             marker=marker, label=v)
ax2.set_xscale("log", base=2)
ax2.set_xticks([32, 64, 128, 256, 512, 1024], ["32", "64", "128", "256", "512", "1024"])
ax2.set_xlabel("threads per block")
ax2.set_ylabel("GPU kernel time per FFT [ms]")
ax2.set_title("Block-size sweep", fontsize=11)
ax2.legend(fontsize=8.5, frameon=False)

fig.suptitle("Double-complex 1D FFT, N = 2^26, NVIDIA V100 (Nsight Systems kernel totals)", fontsize=11.5)
fig.tight_layout()
fig.savefig(here / "fft_versions_v100.png", dpi=150)
print("wrote", here / "fft_versions_v100.png")
