"""Summarize completed token readout runs without further inference."""

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.range_generalization import sha, write


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


KINDS = ("explanatory", "predictive", "counterfactual")


def paired_video_metrics(first, second, draws=10000, seed=17):
    if not first or len(first) != len(second) or len({r["id"] for r in first}) != len(first):
        raise ValueError("empty, duplicate or unequal predictions")
    scenes = sorted({r["scene_index"] for r in first})
    if len(scenes) < 2 or draws < 100:
        raise ValueError("at least two scenes and100 bootstrap draws required")
    index = {s: i for i, s in enumerate(scenes)}
    counts = np.zeros((len(scenes), 3))
    differences = np.zeros((len(scenes), 3, 3))
    for a, b in zip(first, second, strict=True):
        fields = ("id", "scene_index", "question_type", "parent_question_id", "target_index")
        if any(a[k] != b[k] for k in fields) or a["question_type"] not in KINDS:
            raise ValueError("paired prediction identity mismatch")
        values = []
        for row in (a, b):
            p = np.asarray(row["probabilities"], dtype=float)
            target = row["target_index"]
            if (
                p.shape != (2,)
                or not np.isfinite(p).all()
                or (p < 0).any()
                or abs(p.sum() - 1) > 1e-5
                or target not in (0, 1)
                or int(p.argmax()) != row["prediction_index"]
            ):
                raise ValueError("invalid saved probabilities")
            values.append(
                np.array(
                    [
                        float(p.argmax() == target),
                        -np.log(max(p[target], 1e-12)),
                        ((p - np.eye(2)[target]) ** 2).sum(),
                    ]
                )
            )
        i, j = index[a["scene_index"]], KINDS.index(a["question_type"])
        counts[i, j] += 1
        differences[i, j] += values[1] - values[0]
    if (counts.sum(0) == 0).any():
        raise ValueError("missing question type")
    indices = np.random.default_rng(seed).integers(len(scenes), size=(draws, len(scenes)))
    denominators = counts[indices].sum(1)
    if (denominators == 0).any():
        raise ValueError("bootstrap replicate misses question type")
    boot = (differences[indices].sum(1) / denominators[..., None]).mean(1)
    point = (differences.sum(0) / counts.sum(0)[:, None]).mean(0)
    return dict(
        scenes=len(scenes),
        questions=len(first),
        draws=draws,
        seed=seed,
        difference_direction="second minus first; equal question-type macro",
        nll_probability_floor=1e-12,
        metrics={
            name: dict(
                difference=float(point[i]),
                percentile_95_interval=np.quantile(boot[:, i], [0.025, 0.975]).tolist(),
            )
            for i, name in enumerate(("accuracy", "nll", "brier"))
        },
    )


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2/shared-text-video-01")
    result, config = read(root / "result.json"), read(root / "config.json")
    assert result["status"] == "PASS_EXECUTION_ONLY" and not result["smoke"]
    histories = {name: read(root / name / "history.json") for name in result["conditions"]}
    assert all(len(h) == 24 for h in histories.values())
    extraction = read(root / "extraction.json")
    assert len(extraction) == 456 and len({r["media_sha256"] for r in extraction}) == 456
    assert max(r["pooled_max_abs"] for r in extraction) <= 1e-5
    predictions = {}
    for name, r in result["conditions"].items():
        for suffix in ("", "_shuffled"):
            predictions[name + suffix] = read(root / name / f"predictions{suffix}.json")
        assert r["reload_exact"] and r["consumed"] == 91056
        assert (
            r["best_epoch"]
            == min(histories[name], key=lambda x: x["validation"]["macro_question_type"]["nll"])[
                "epoch"
            ]
        )
        for label in ("best", "final"):
            assert sha(root / name / f"{label}.pt") == r[f"{label}_sha256"]
    for name in ("minilm_no_video", "shared_no_video"):
        for a, b in zip(predictions[name], predictions[name + "_shuffled"], strict=True):
            assert a["id"] == b["id"] and a["probabilities"] == b["probabilities"]
    assert len({v["initial_tensor_sha256"] for v in result["conditions"].values()}) == 1
    assert len({v["order_sha256"] for v in result["conditions"].values()}) == 1
    reference = read(root.parent / "video-token-readout-01/extraction.json")
    assert {r["media_sha256"]: r["token_sha256"] for r in extraction} == {
        r["media_sha256"]: r["token_sha256"] for r in reference
    }
    pairs = [
        ("shared_minus_minilm_video", "minilm_attention", "shared_attention"),
        ("shared_minus_minilm_no_video", "minilm_no_video", "shared_no_video"),
        ("minilm_video_minus_no_video", "minilm_no_video", "minilm_attention"),
        ("shared_video_minus_no_video", "shared_no_video", "shared_attention"),
        ("minilm_matched_minus_shuffled", "minilm_attention_shuffled", "minilm_attention"),
        ("shared_matched_minus_shuffled", "shared_attention_shuffled", "shared_attention"),
    ]
    metric_comparisons = {
        name: paired_video_metrics(predictions[a], predictions[b]) for name, a, b in pairs
    }
    for name, value in metric_comparisons.items():
        assert (
            abs(
                value["metrics"]["accuracy"]["difference"]
                - result["comparisons"][name]["difference"]
            )
            < 1e-12
        )
    config["id_provenance"] = {
        split: dict(
            count=len(ids),
            sha256=hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        )
        for split, ids in config.pop("ids").items()
    }
    out = Path("docs/pretrained_reuse/R2_SHARED_TEXT_VIDEO_RESULTS_20261012")
    write(
        out.with_suffix(".json"),
        dict(
            result=result,
            metric_comparisons=metric_comparisons,
            config=config,
            histories=histories,
            hashes={str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*.json"))},
        ),
    )
    lines = [
        "# Shared text features with matched video/no-video readouts",
        "",
        "Execution completed; previously observed development validation, not blind quality.",
        f"Source `{config['source_commit']}`. Seed17; 24epochs each; fixed split and order.",
        "3794 unique training choices/256 scenes; 3014 validation choices/200 scenes.",
        "Each condition sees 91,056 encounters (87,262 repeats). No sealed audit access.",
        "456 full FP32 token tensors extracted transiently; every pooled mean matches prior cache.",
        "",
        "| Mode | Type | Accuracy | NLL | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for mode, r in result["conditions"].items():
        for kind, m in {
            "macro": r["matched"]["macro_question_type"],
            **r["matched"]["by_question_type"],
        }.items():
            lines.append(
                f"| {mode} | {kind} | {m['accuracy']:.6f} | {m['nll']:.6f} | "
                f"{m['brier']:.6f} | {m['ece_15_bins']:.6f} |"
            )
    lines += ["", "## Paired scene uncertainty", ""]
    for name, value in result["comparisons"].items():
        low, high = value["percentile_95_interval"]
        lines.append(
            f"- {name}: {value['difference'] * 100:+.3f}pp; "
            f"95% interval [{low * 100:+.3f}, {high * 100:+.3f}]pp."
        )
    lines += [
        "",
        "200-scene paired percentile bootstrap, 10,000 draws, seed17;",
        "intervals do not remove validation selection bias or multiple comparisons.",
        "",
        "## Learning curves",
        "",
        "| Mode | Epoch | Train Acc | Val Acc | Train NLL | Val NLL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, history in histories.items():
        for row in history:
            t, v = row["train"]["macro_question_type"], row["validation"]["macro_question_type"]
            lines.append(
                f"| {mode} | {row['epoch']} | {t['accuracy']:.6f} | {v['accuracy']:.6f} | "
                f"{t['nll']:.6f} | {v['nll']:.6f} |"
            )
    lines += ["", "## Resources and artifacts", ""]
    for mode, r in result["conditions"].items():
        lines.append(
            f"- {mode}: best epoch {r['best_epoch']}, selected encounters "
            f"{r['selected_consumed']}, total parameters {r['parameters']}, "
            f"active parameters {r['active_parameters']}, {r['seconds']:.3f}s."
        )
    lines += [
        f"- Extraction {result['extraction_seconds']:.3f}s; "
        f"total {result['total_seconds']:.3f}s; peak allocated VRAM "
        f"{result['peak_vram_bytes']} bytes; local cost $0.",
        "- Frozen video:114,365,184; text: MiniLM17,389,824 or shared Cross-Encoder22,713,601.",
        "  Mean/zero-video controls carry unused key/position parameters; active capacity differs.",
        "- Best/final weights separate; best reload exact. Original artifacts preserved.",
        f"- Artifacts `{root.as_posix()}`; full probabilities, histories and hashes retained.",
        "- smoke-01 failed on worker imports; smoke-02 passed all four conditions.",
        "- Four-modality95% and general decision quality remain unproven.",
    ]
    lines += ["", "## Probability-score paired comparisons", ""]
    for name, c in metric_comparisons.items():
        for metric in ("nll", "brier"):
            value = c["metrics"][metric]
            lo, hi = value["percentile_95_interval"]
            lines.append(
                f"- {name}, {metric}: {value['difference']:+.6f}; "
                f"95% interval [{lo:+.6f}, {hi:+.6f}]."
            )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "No automatic adoption or additional epochs. No-video has fewer functionally",
        "influential parameters. These controls measure video contribution.",
        "Macro weights question types equally, and bootstrap resamples whole scenes.",
        "A weak-baseline match does not establish useful temporal reasoning or a capacity ceiling.",
        "The combined parameter budget requires one actually shared loaded text tower;",
        "this separate video experiment alone is not an assembled <=200M Omni runtime.",
        "No sealed audit, new seed search or quantization was performed.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
