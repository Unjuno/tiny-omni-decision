"""Verify and summarize the completed descriptive token comparison."""

from pathlib import Path

from scripts.range_generalization import sha, write
from scripts.report_video_grounding import compare, read


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2/grounding-token-training-01")
    result, config = read(root / "result.json"), read(root / "config.json")
    assert result["status"] == "PASS_EXECUTION_ONLY" and not result["smoke"]
    extraction = read(root / "extraction.json")
    assert len(extraction) == 456 and len({r["media_sha256"] for r in extraction}) == 456
    assert max(r["pooled_max_abs"] for r in extraction) <= 1e-5
    histories = {}
    predictions = {}
    for mode, r in result["conditions"].items():
        history = read(root / mode / "history.json")
        assert len(history) == 25 and r["reload_exact"] and r["epochs"] == 24
        assert (
            r["best_epoch"]
            == min(history, key=lambda h: h["validation"]["macro_taxonomy"]["nll"])["epoch"]
        )
        for label in ("best", "final"):
            assert sha(root / mode / f"{label}.pt") == r[f"{label}_sha256"]
        histories[mode] = history
        for kind in ("matched", "shuffled"):
            predictions[mode + "_" + kind] = read(root / mode / f"{kind}-predictions.json")
    assert predictions["no_video_matched"] == predictions["no_video_shuffled"]
    pairs = {
        "attention_minus_mean": ("mean_matched", "attention_matched"),
        "attention_minus_no_video": ("no_video_matched", "attention_matched"),
        "mean_minus_no_video": ("no_video_matched", "mean_matched"),
        "attention_matched_minus_shuffled": ("attention_shuffled", "attention_matched"),
        "mean_matched_minus_shuffled": ("mean_shuffled", "mean_matched"),
    }
    comparisons = {name: compare(predictions[a], predictions[b]) for name, (a, b) in pairs.items()}
    output = Path("docs/pretrained_reuse/R2_GROUNDING_TOKEN_RESULTS_20261012.json")
    if output.exists():
        raise FileExistsError(output)
    write(
        output,
        dict(
            result=result,
            config=config,
            histories=histories,
            comparisons=comparisons,
            bootstrap=dict(unit="scene", draws=10000, seed=17, aggregation="equal taxonomy"),
            hashes={str(p): sha(p) for p in root.rglob("*") if p.is_file()},
        ),
    )
    for name, r in comparisons.items():
        print(name, r)


if __name__ == "__main__":
    main()
