"""Plot the frozen development label mix without media/model inference."""

import importlib.metadata
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.probe_greatest_hits_train import sha


def main():
    source = Path("C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-train-inventory-01")
    path = source / "inventory.json"
    if sha(path) != "3dab5814a6a5e87599f5d67ad435d7f8a81042e426969c004b9eed6e3d74a19d":
        raise ValueError("frozen inventory mismatch")
    data = json.loads(path.read_text())
    output = Path("C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-development-label-plot-01")
    output.mkdir(exist_ok=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 7), constrained_layout=True)
    for field, (ax, task) in enumerate(zip(axes, ["Material", "Action", "Reaction"], strict=True)):
        labels = data["task_vocabularies"][field]
        for split, offset, color, description in [
            ("train", -0.18, "#315e8c", "TRAIN / 308 parents"),
            ("validation", 0.18, "#c47227", "Development validation / 101 parents"),
        ]:
            counts = data["splits"][split]["fields"][field]
            known = sum(n for label, n in counts.items() if label != "<MISSING>")
            ax.barh([i + offset for i in range(len(labels))],
                    [100 * counts.get(label, 0) / known for label in labels],
                    height=0.34, color=color, label=description)
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_title(task + f" ({len(labels)} options)")
        ax.set_xlabel("Share of known labels (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Greatest Hits: fixed date-grouped development label distributions", fontsize=14)
    fig.supxlabel("Missing labels are counted separately; no date/class rebalancing. "
                  "Official TEST media and labels remain closed.", fontsize=10)
    fig.savefig(output / "label-distribution.png", dpi=160)
    plt.close(fig)
    receipt = dict(source=str(path), source_sha256=sha(path), script_sha256=sha(Path(__file__)),
                   output_sha256=sha(
        output / "label-distribution.png"), versions={p: importlib.metadata.version(p)
                                                    for p in ("numpy", "matplotlib")},
                   model_inference=False)
    (output / "result.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
