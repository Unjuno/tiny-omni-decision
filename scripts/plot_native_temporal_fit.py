"""Render measured fixed learning curves; no model inference or recomputed labels."""

import importlib.metadata
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("C:/CodexArtifacts/pretrained-reuse-r2/native-common-comparison-01")
r = json.loads((root / "comparison.json").read_text())
assert r["status"] == "PASS_PAIRED_COMPARISON_AUDIT"
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160}
)
fig, axs = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
colors = {"baseline": "#56647A", "lstm": "#167C80"}
for name, color in colors.items():
    curve = r["curves"][name]
    x = [h["step"] for h in curve]
    axs[0, 0].plot(
        x,
        [h["validation"]["macro"]["nll"] for h in curve],
        color=color,
        label=name + " validation",
        marker="o",
        ms=3,
    )
    axs[0, 0].plot(
        x,
        [h["train_probe"]["macro"]["nll"] for h in curve],
        color=color,
        ls="--",
        label=name + " TRAIN probe",
    )
    for task, style in [("temporal_descriptive", "-"), ("static_descriptive", "--")]:
        axs[0, 1].plot(
            x,
            [100 * h["validation"]["tasks"]["clevrer:" + task]["accuracy"] for h in curve],
            color=color,
            ls=style,
            label=name + " " + task.split("_")[0],
            marker="o",
            ms=3,
        )
ax = axs[0, 0]
ax.set_title("Fixed learning curve: macro NLL")
ax.set_xlabel("Optimizer updates")
ax.set_ylabel("NLL (lower is better)")
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
ax = axs[0, 1]
ax.set_title("Video development questions")
ax.set_xlabel("Optimizer updates")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)
modalities = ["text", "image", "audio", "video"]
x = np.arange(4)
for i, (name, color) in enumerate(colors.items()):
    final = r["runs"][name]["final_metrics"]
    vals = [100 * final["modality_macro"][m]["accuracy"] for m in modalities]
    bars = axs[1, 0].bar(x + (i - 0.5) * 0.35, vals, width=0.35, label=name, color=color)
    axs[1, 0].bar_label(bars, fmt="%.1f", fontsize=8)
    speech = 100 * final["source_macro"]["lombard"]["accuracy"]
    events = 100 * final["source_macro"]["starss"]["accuracy"]
    recalls = [v["recall"] for v in final["event_classes"].values() if v["recall"] is not None]
    vals = [speech, events, 100 * np.mean(recalls)]
    bars = axs[1, 1].bar(np.arange(3) + (i - 0.5) * 0.35, vals, width=0.35, label=name, color=color)
    axs[1, 1].bar_label(bars, fmt="%.1f", fontsize=8)
ax = axs[1, 0]
ax.set_title("Fixed2048 endpoint: four modalities")
ax.set_xticks(x, modalities)
ax.set_ylim(0, 105)
ax.set_ylabel("Task/source macro Accuracy (%)")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.2)
ax = axs[1, 1]
ax.set_title("Audio Accuracy versus positive detection")
ax.set_xticks(np.arange(3), ["Speech Accuracy", "Event Accuracy", "Event positive recall"])
ax.set_ylim(0, 105)
ax.set_ylabel("Percent")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.2)
fig.suptitle(
    "Native Decision readout: attention baseline vs residual LSTM\n"
    "Seed17, identical data/order and2048 updates; development validation, not blind audit",
    fontsize=14,
)
fig.supxlabel(
    "Frozen encoders; head-only training. TRAIN probe is mostly unconsumed examples. "
    "All-task95-98% goal remains unmet.",
    fontsize=9,
)
fig.savefig(root / "learning-curves.png")
fig.savefig(root / "learning-curves.svg")
(root / "plot-environment.json").write_text(
    json.dumps(
        {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "backend": matplotlib.get_backend(),
            "separate_dependencies": str(root.parent / "native-comparison-plot-deps-01"),
            "training_environment_changed": False,
            "separate_package_versions": {
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions(
                    path=[str(root.parent / "native-comparison-plot-deps-01")]
                )
            },
        },
        indent=2,
    )
    + "\n"
)
