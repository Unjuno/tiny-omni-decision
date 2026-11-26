"""Plot the measured full-extent/control learning curves without model inference."""

import importlib.metadata
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("C:/CodexArtifacts/pretrained-reuse-r2/native-full-extent-comparison-01")
r = json.loads((OUT / "comparison.json").read_text())
assert r["status"] == "PASS_FULL_EXTENT_EXECUTION_AND_REPLAY_AUDIT"
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160}
)
fig, axs = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
series = (
    (r["control_curve"], "validation", "Control validation", "#68758b", "-"),
    (r["curve"], "validation", "Full extent validation", "#087f83", "-"),
    (r["curve"], "train_probe", "Full extent fixed TRAIN", "#d28b2c", "--"),
)
for ax, mod in zip(axs.flat, ("text", "image", "audio", "video"), strict=False):
    for curve, key, label, color, style in series:
        ax.plot(
            [h["step"] for h in curve],
            [100 * h[key]["modality_macro"][mod]["accuracy"] for h in curve],
            style,
            color=color,
            marker="o",
            ms=3,
            label=label,
        )
    ax.set_title(mod.title() + ": task/source macro Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
for curve, key, label, color, style in series:
    x = [h["step"] for h in curve]
    axs[2, 0].plot(
        x,
        [h[key]["macro"]["nll"] for h in curve],
        style,
        color=color,
        marker="o",
        ms=3,
        label=label,
    )
    axs[2, 1].plot(
        x,
        [100 * h[key]["tasks"]["clevrer:temporal_descriptive"]["accuracy"] for h in curve],
        style,
        color=color,
        marker="o",
        ms=3,
        label=label,
    )
axs[2, 0].set_title("Task/source/modality macro NLL")
axs[2, 0].set_ylabel("NLL (lower is better)")
axs[2, 1].set_title("Temporal Video Accuracy")
axs[2, 1].set_ylabel("Accuracy (%)")
axs[2, 1].set_ylim(0, 100)
for ax in axs[2]:
    ax.legend(fontsize=8)
for ax in axs.flat:
    ax.set_xscale("symlog", base=2, linthresh=512)
    ax.set_xticks([0, 512, 2048, 8192, 32768], ["0", "512", "2,048", "8,192", "32,768"])
    ax.set_xlabel("Optimizer updates (log scale above512)")
    ax.grid(alpha=0.2)
fig.suptitle(
    "Full visual extent vs center crop: fixed32,768-update comparison\n"
    "Same seed17, sample order, initialization, model and LR schedule",
    fontsize=15,
)
fig.supxlabel(
    "Observed development validation; TRAIN/validation task mixes differ.\n"
    "Full extent changes object scale and padding as well as spatial coverage.",
    fontsize=9,
)
fig.savefig(OUT / "learning-curves.png")
plt.close(fig)
(OUT / "plot-environment.json").write_text(
    json.dumps({p: importlib.metadata.version(p) for p in ("numpy", "matplotlib")}, indent=2) + "\n"
)
