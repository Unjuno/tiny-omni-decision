"""Summarize the executed two-scene token audit and retained failed attempt."""

from datetime import datetime
from pathlib import Path

from scripts.range_generalization import sha, write
from scripts.report_spatial_curve import read


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    run = root / "videoprism-token-audit-02"
    result, config = (read(run / f"{name}.json") for name in ["result", "config"])
    assert result["status"] == "PASS_FEASIBILITY_ONLY" and result["repeat_exact"]
    elapsed = (
        datetime.fromisoformat(result["finished_utc"])
        - datetime.fromisoformat(config["started_utc"])
    ).total_seconds()
    old = root / "videoprism-token-audit-01"
    assert not (old / "result.json").exists()
    for index in range(2):
        assert sha(old / f"tokens-{index}.npy") == sha(run / f"tokens-{index}.npy")
    out = Path("docs/pretrained_reuse/R2_VIDEO_TOKEN_AUDIT_RESULTS_20261012")
    write(
        out.with_suffix(".json"),
        dict(
            config=config,
            result=result,
            elapsed_from_config_seconds=elapsed,
            hashes={p.name: sha(p) for p in sorted(run.iterdir()) if p.is_file()},
            failed_attempt=dict(
                config=read(old / "config.json"),
                progress=read(old / "progress.json"),
                error="importlib.metadata.PackageNotFoundError: tensorflow",
                total_seconds="UNKNOWN: final reporting failed",
                raw_arrays_match_retry=True,
                hashes={p.name: sha(p) for p in sorted(old.iterdir()) if p.is_file()},
            ),
        ),
    )
    lines = [
        "# Unpooled VideoPrism token feasibility result",
        "",
        "PASS_FEASIBILITY_ONLY on two preselected training scenes19/37. No training,",
        "validation scoring, final audit, or claim of better temporal reasoning.",
        f"Source `{config['source_commit']}`; exact command/model/source/media hashes",
        "and versions in companion JSON. Existing upstream tracked source unchanged;",
        "pre-existing build/egg-info/pycache directories were left in place.",
        "",
        "## Measured representation",
        "",
        "Pinned VideoPrism-B has114,365,184 parameters. At the unchanged8frames and",
        "288×288 preprocessing it returns FP32[1,2048,768], exactly reshaped as",
        "[8,16,16,768]. Upstream `(bn)td -> b(tn)d` establishes time-major indexing;",
        "the index mapping is unit-tested. No pooling or precision conversion in saved arrays.",
        "",
        "Both arrays reload exactly. Their global means equal the existing pooled",
        "reference with maximum difference0.0. Repeating the first forward returns",
        "exactly identical tokens. The saved arrays also match the first attempt's hashes.",
        "These are contextualized grid tokens, not verified object tracks.",
        "",
        "## Runtime and storage",
        "",
        f"- Backend{result['backend']}, device{result['devices']}; no GPU execution claim.",
        f"- Model load{result['load_seconds']:.3f}s; first forward including compilation"
        f" {result['records'][0]['forward_seconds']:.3f}s.",
        f"- Second scene forward{result['records'][1]['forward_seconds']:.3f}s; "
        f"repeat forward{result['repeat_forward_seconds']:.3f}s.",
        f"- Model/audit timer{result['total_seconds']:.3f}s excludes dependency imports.",
        f"  Config-start to finish{elapsed:.3f}s includes those imports but excludes",
        "  interpreter startup and earlier provenance checks; not a full CLI benchmark.",
        f"- Windows peak working set{result['memory']['peak_wset']:,}bytes; "
        f"final RSS{result['memory']['rss']:,}bytes.",
        f"- Each raw array{result['records'][0]['array_bytes']:,}bytes; "
        f"NPY file{result['records'][0]['file_bytes']:,}bytes.",
        f"- Projected456-scene NPY footprint{result['projected456_scene_npy_bytes']:,}bytes",
        f"  exceeds observed free disk{result['free_disk_bytes']:,}bytes. This projection",
        "  is not an executed full extraction. No full cache was started.",
        "- Local compute only,cloud cost$0. Only two training assets were read.",
        "",
        "## Failed attempt and correction",
        "",
        "Attempt01 extracted both arrays and passed the earlier checks, but failed",
        "while writing final metadata because this Windows installation does not",
        "provide a distribution named tensorflow. Its total/repeat timings were not",
        "persisted and remain UNKNOWN. Partial outputs were retained unchanged.",
        "Use the imported tf.__version__ instead of assuming a distribution alias;",
        "regression test covers this case. No packages were installed or upgraded.",
        "Attempt02 ran afresh in a new directory and completed all gates.",
        "",
        "## Next bounded step",
        "",
        "Token extraction is practical in this tiny audit; quality is unmeasured.",
        "A full FP32 file cache does not fit current free disk. Prefer a measured",
        "bounded transfer to a RAM cache shared by all questions for each scene,",
        "rather than duplicating tokens per question or silently lowering precision.",
        "Measure that memory/transport path before a full token-readout experiment.",
        "Keep frame count and encoders fixed; no basis yet for full fine-tuning.",
        "",
        "Artifacts remain at `C:/CodexArtifacts/pretrained-reuse-r2/videoprism-token-audit-01`",
        "and `videoprism-token-audit-02`. Large arrays stay outside Git. Four-modality",
        "95–98% remains unmet; this audit supplies an execution path, not quality evidence.",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
