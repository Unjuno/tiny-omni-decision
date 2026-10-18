"""Draw full fixed-budget curves and all event-class retention results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2/native-support-comparison-01")
data = json.loads((ROOT / "comparison.json").read_text(encoding="utf-8"))
assert data["status"] == "PASS_SUPPORT_EXECUTION_AND_REPLAY_AUDIT"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
colors = {"Control": "#64748b", "Positive support": "#007f89"}
curves = [("Control", data["parent_curve"]), ("Positive support", data["curve"])]
for label, curve in curves:
    steps = [h["step"] for h in curve]
    color = colors[label]
    for ax, path, title in (
        (axes[0, 0], ("macro", "nll"), "Validation macro NLL (lower is better)"),
        (axes[0, 1], ("macro", "accuracy"), "Validation macro accuracy"),
        (axes[0, 2], ("source_macro", "starss", "accuracy"), "STARSS accuracy"),
        (axes[1, 0], ("source_macro", "lombard", "accuracy"), "Lombard accuracy"),
    ):
        values = []
        for h in curve:
            value = h["validation"]
            for key in path:
                value = value[key]
            values.append(value)
        ax.plot(steps, values, "o-", ms=4, color=color, label=label)
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Optimizer updates")
    balanced = []
    for h in curve:
        ev = h["validation"]["event_classes"]
        values = [
            v["balanced_accuracy"]
            for k, v in ev.items()
            if k != "speech" and v["balanced_accuracy"] is not None
        ]
        balanced.append(np.mean(values))
    axes[1, 1].plot(steps, balanced, "o-", ms=4, color=color, label=label)
axes[1, 1].set(title="Non-speech macro balanced accuracy", xlabel="Optimizer updates")
axes[1, 1].grid(alpha=0.2)
axes[0, 0].legend(frameon=False)
for ax in [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]:
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
ax = axes[1, 2]
mods = ("text", "image", "audio", "video")
x = np.arange(4)
for i, key in enumerate(("control", "candidate")):
    values = [data["comparisons"]["selected"][key]["modality_macro"][m]["accuracy"] for m in mods]
    ax.bar(
        x + (i - 0.5) * 0.36,
        values,
        0.36,
        color=colors["Control" if i == 0 else "Positive support"],
    )
ax.set(xticks=x, xticklabels=mods, ylim=(0, 1), title="Selected checkpoints: modality accuracy")
ax.axhline(0.95, color="#b91c1c", ls="--", lw=1)
ax.text(3.4, 0.955, "95% target", color="#b91c1c", ha="right", fontsize=9)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
fig.suptitle(
    "One sampling intervention, unchanged 32,768-update budget\n"
    "Development validation; 292 unique negative questions replaced by unused positives",
    fontsize=14,
)
fig.savefig(ROOT / "learning-curves.png", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.5), constrained_layout=True)
events = data["comparisons"]["selected"]["control"]["event_classes"]
names = list(events)
x = np.arange(len(names))
for ax, metric in zip(axes, ("recall", "specificity"), strict=True):
    for i, key in enumerate(("control", "candidate")):
        ev = data["comparisons"]["selected"][key]["event_classes"]
        values = [np.nan if ev[n][metric] is None else ev[n][metric] for n in names]
        ax.bar(
            x + (i - 0.5) * 0.36,
            values,
            0.36,
            color=colors["Control" if i == 0 else "Positive support"],
            label="Control" if i == 0 else "Positive support",
        )
    ax.set(
        ylim=(0, 1.12),
        title="Selected checkpoint: " + metric,
        xticks=x,
        xticklabels=[
            n + f"\n+{events[n]['positive_count']}/-{events[n]['negative_count']}" for n in names
        ],
    )
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(axis="y", alpha=0.2)
axes[0].legend(frameon=False)
fig.suptitle(
    "All 12 STARSS classes retained\n"
    "Bell has no admitted TRAIN positives; water-tap validation recall is undefined",
    fontsize=13,
)
fig.savefig(ROOT / "event-retention.png", dpi=160)
plt.close(fig)
