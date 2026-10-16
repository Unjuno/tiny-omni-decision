"""Render actual fixed TRAIN/development learning curves, without model inference."""

import importlib.metadata
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("C:/CodexArtifacts/pretrained-reuse-r2/native-long-budget-comparison-01")
r = json.loads((OUT / "comparison.json").read_text())
assert r["status"] == "PASS_LONG_CURVE_EXECUTION_AND_REPLAY_AUDIT"
c = r["curve"]
x = [h["step"] for h in c]
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160}
)
fig, axs = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
for ax, mod in zip(axs.flat, ("text", "image", "audio", "video"), strict=False):
    for key, style, label, color in (
        ("train_probe", "--", "Fixed TRAIN pool", "#57677c"),
        ("validation", "-", "Development validation", "#127f83"),
    ):
        ax.plot(
            x,
            [100 * h[key]["modality_macro"][mod]["accuracy"] for h in c],
            style,
            marker="o",
            ms=3,
            color=color,
            label=label,
        )
    ax.set_title(mod.title() + ": task/source macro Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
ax = axs[2, 0]
for key, style, color in (("train_probe", "--", "#57677c"), ("validation", "-", "#127f83")):
    ax.plot(
        x,
        [h[key]["macro"]["nll"] for h in c],
        style,
        color=color,
        marker="o",
        ms=3,
        label=key.replace("train_probe", "fixed TRAIN"),
    )
ax.set_title("Complete task/source/modality macro NLL")
ax.set_ylabel("NLL (lower is better)")
ax.legend(fontsize=8)
ax = axs[2, 1]
for key, style, color in (("train_probe", "--", "#57677c"), ("validation", "-", "#127f83")):
    ax.plot(
        x,
        [100 * h[key]["tasks"]["clevrer:temporal_descriptive"]["accuracy"] for h in c],
        style,
        color=color,
        marker="o",
        ms=3,
        label=key.replace("train_probe", "fixed TRAIN"),
    )
ax.set_title("Temporal Video Accuracy")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.legend(fontsize=8)
for ax in axs.flat:
    ax.axvline(2048, color="#b97629", ls=":", lw=1)
    ax.set_xscale("symlog", base=2, linthresh=512)
    ax.set_xticks([0, 512, 2048, 8192, 32768], ["0", "512", "2,048", "8,192", "32,768"])
    ax.set_xlabel("Optimizer updates (log scale above512)")
    ax.grid(alpha=0.2)
fig.suptitle(
    "Native Decision: predeclared32,768-update curve\n"
    "Seed17, unchanged6,656-question pool and196.25M total parameters",
    fontsize=15,
)
fig.supxlabel(
    "TRAIN/validation task mixes differ; common43-task gaps are saved separately.\n"
    "Observed development data, not blind audit. No performance early stopping; "
    "separate best/final checkpoints.",
    fontsize=9,
)
fig.savefig(OUT / "learning-curves.png")
fig.savefig(OUT / "learning-curves.svg")
plt.close(fig)

fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
s = r["segments"]
sx = [h["end"] for h in s]
axs[0, 0].plot(
    x, [np.sqrt(sum(v * v for v in h["parameter_l2_norms"].values())) for h in c], marker="o"
)
axs[0, 0].set_title("Readout total parameter L2 norm")
axs[0, 1].plot(sx, [h["median_gradient_norm"] for h in s], marker="o")
axs[0, 1].set_title("Median gradient norm before global clipping")
axs[1, 0].plot(sx, [100 * h["clipped_fraction"] for h in s], marker="o")
axs[1, 0].set_title("Updates exceeding clip1 (%)")
axs[1, 0].set_ylim(0, 100)
for key, style in (("train", "--"), ("validation", "-")):
    axs[1, 1].plot(
        x, [h["common_task_gap"][key]["macro"]["nll"] for h in c], style, marker="o", label=key
    )
axs[1, 1].set_title("Only43 shared tasks: macro NLL")
axs[1, 1].legend()
for ax in axs.flat:
    ax.set_xlabel("Optimizer updates")
    ax.grid(alpha=0.2)
fig.suptitle("Measured optimization diagnostics; no claim of a grokking mechanism")
fig.savefig(OUT / "optimization-diagnostics.png")
plt.close(fig)
(OUT / "plot-environment.json").write_text(
    json.dumps({p: importlib.metadata.version(p) for p in ("numpy", "matplotlib")}, indent=2) + "\n"
)
