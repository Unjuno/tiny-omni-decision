"""Publish verified results; preserve the executed source and original render attempt."""

import copy
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
OUT = ROOT / "native-full-extent-comparison-01"
DOCS = Path(__file__).resolve().parents[1] / "docs/pretrained_reuse"
STEM = "R2_NATIVE_FULL_EXTENT_RESULTS_20261016"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def comparison_row(label, delta):
    cells = [label, f"{100 * delta['macro']['accuracy']:.3f}", f"{delta['macro']['nll']:.6f}"]
    cells.extend(
        f"{100 * delta['modalities'][mod]['accuracy']:.3f}"
        for mod in ("text", "image", "audio", "video")
    )
    return "| " + " | ".join(cells) + " |"


def main():
    value = json.loads((OUT / "comparison.json").read_text(encoding="utf8"))
    assert value["status"] == "PASS_FULL_EXTENT_EXECUTION_AND_REPLAY_AUDIT"
    assert value["run"]["best_step"] == 12288
    assert not value["retention"]["retain_for_research"]
    text = (OUT / "report.md").read_text(encoding="utf8")
    start = text.index("### Equal-step differences")
    end = text.index("### Paired uncertainty")
    table = [
        "### Selected and equal-step differences (full extent minus control)",
        "",
        "| Comparison | Macro Acc (pp) | Macro NLL | Text Acc (pp) | "
        "Image Acc (pp) | Audio Acc (pp) | Video Acc (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    table.extend(
        comparison_row(label, item["deltas"]) for label, item in value["comparisons"].items()
    )
    text = text[:start] + "\n".join(table) + "\n\n" + text[end:]
    text = text.replace("**False**.\n-", "**False**.\n\n-")
    summary = """## Measured interpretation

Executed on2026-10-16–17 JST. Preserve this candidate as a completed negative
research result; do not replace the original control or promote a product.

- Selected Image Accuracy35.4004→38.1348%, NLL1.798945→1.707391 and
  Brier0.735258→0.708370 improve. Its paired image Accuracy difference is
  +2.734pp, diagnostic95% interval[+0.977,+4.541].
- Selected Video Accuracy33.75→38.00%, but its interval[−0.75,+9.25]pp includes0.
  Video NLL worsens1.501657→1.691322. Temporal Accuracy's interval also includes0.
- Audio Accuracy61.2724→58.1766% declines3.096pp. Its NLL/ECE improve slightly;
  record the mixed trade-off rather than calling every Audio metric worse.
- Macro Accuracy37.4174→38.7589% improves while macro NLL1.630249→1.673511,
  Brier0.705674→0.717432 and ECE0.126081→0.135351 worsen. The predefined
  macro-NLL and nonvisual-retention checks fail.
- Equal8192-step macro Accuracy/NLL are both worse. At equal32768, NLL is lower
  but macro Accuracy is4.015pp lower; do not use endpoint NLL alone as success.
- After selected12288, TRAIN macro NLL falls to0.662113 while validation NLL
  rises to3.206027. Final Video TRAIN99.805% versus validation33.75%, and
  temporal TRAIN100% versus validation36%, show a large memorization gap.
  Full TRAIN/validation task mixes differ;43 shared-task results remain available.
  Overfitting warnings occur at8192/16384/24576/32768. These observations do not
  establish a general capability ceiling or rule out delayed learning elsewhere.

The audit identified real restrictions, but this one full-extent policy did not
solve the pipeline's quality problem. Restored extent, smaller object scale,
padding and joint optimization effects are not separately identified. The source
labels and all difficult validation examples remain unchanged.

### Next bounded question

The previous unique-coverage comparison ended at2048 updates, while longer curves
show continued changes. Next preregister one matched32768-update comparison using
the already-frozen fresh TRAIN pool against the original repeat control: same
original crop/features, model, seed17, LR schedule, validation IDs and selector;
only the TRAIN coverage policy changes. Keep the failed support and full-extent
policies out of that comparison. Record unique assets and class support, preserve
Video's limited256-scene coverage, and keep all modality trade-offs. This is not
yet run or scheduled by this report. Temporal/LSTM/xLSTM hypotheses remain open;
this result neither validates nor rules them out. No geometry/seed search or audit.

"""
    marker = "## Intervention and integrity"
    text = text.replace(marker, summary + marker, 1)
    text += """
## Verification and publication provenance

Training and fresh replay used immutable source `ce705f8`; its CI#1847 passed.
Pretraining Ruff,2170 CPU tests (1 skip) and25 manifest commands passed. All32
initial smoke/main updates have exactly matching CE/Accuracy/LR/gradient norms,
sample IDs and occurrence IDs. Final publication verification is recorded under
`native-full-extent-verification-01/publication/` and reported with the pushed commit.

The first report render had a string-join precedence error in its comparison
table. Preserve that `report.md` and executed generator; this publication helper
corrects only presentation and has a regression test. No metrics or checkpoints
were changed. The full private `comparison.json` remains immutable. The public JSON
keeps every candidate TRAIN/validation task/source/modality metric; redundant control
curves, endpoint TRAIN diagnostics and duplicate common-task detail are omitted and
referenced by the full result's SHA. Older control details remain in its own report.

Runtime context: Windows11 build26300, i7-12700H (14cores/20threads), driver616.92,
RTX3080 Laptop16GiB. Periodic lightweight CPU checks occurred; timings are observed
wall times, not an isolated hardware benchmark. Local electricity was not metered.
"""
    public = copy.deepcopy(value)
    for key in ("control_curve", "endpoint_state_metrics", "endpoint_train_metrics"):
        public.pop(key)
    for point in public["curve"]:
        point["accounting"].pop("tasks", None)
        for key in ("train", "validation"):
            point["common_task_gap"][key].pop("tasks", None)
    public["full_result"] = dict(
        path=str(OUT / "comparison.json"), sha256=sha(OUT / "comparison.json")
    )
    public["publication_scope"] = (
        "Complete candidate TRAIN/validation task/source/modality metrics; original full "
        "comparison remains local and hashed. No data rows or quality metrics were filtered."
    )
    public["verification"] = {
        name: json.loads(
            (ROOT / "native-full-extent-verification-01" / name).read_text(encoding="utf8")
        )
        for name in (
            "pretraining/status.json",
            "pretraining/ci-source.json",
            "initial-prefix-parity.json",
            "runtime-context.json",
        )
    }
    destinations = [
        OUT / "report-final.md",
        OUT / "public-summary.json",
        DOCS / (STEM + ".md"),
        DOCS / (STEM + ".json"),
        DOCS / (STEM + ".png"),
    ]
    if any(path.exists() for path in destinations):
        raise ValueError("new publication paths required; do not overwrite historical outputs")
    destinations[0].write_text(text, encoding="utf8")
    destinations[1].write_text(
        json.dumps(public, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    for source, target in zip(
        (destinations[0], destinations[1], OUT / "learning-curves.png"),
        destinations[2:],
        strict=True,
    ):
        shutil.copyfile(source, target)
    print("Published verified metrics; research retention remains False")


if __name__ == "__main__":
    main()
