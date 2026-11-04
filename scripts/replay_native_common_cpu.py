"""CPU-only replay of the pinned native common readout on its validation split."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.native_fit import summarize  # noqa: E402
from scripts.native_fit_runtime import FeatureStore  # noqa: E402
from scripts.native_video_cache import sha  # noqa: E402
from tiny_omni_decision.native_candidate import NativeCandidateReadout  # noqa: E402

CONFIG = REPO / "configs/pretrained_reuse/native_common_fit_v1.json"
SOURCE_ROOT = Path(r"C:\CodexWorktrees\pretrained-reuse-omni")
SCORE_ATOL = 5e-5


def score_validation(model, rows, store, progress=None):
    """Score rows in their supplied order, preserving the training evaluator's padding."""
    scores = np.full((len(rows), 40), -1e30, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for index, row in enumerate(rows):
            blocks, queries = store.get(row)
            values = model(blocks, queries).cpu().numpy()
            if values.shape != (len(row["options"]),) or not np.isfinite(values).all():
                raise ValueError("readout returned invalid option scores")
            scores[index, : len(values)] = values
            completed = index + 1
            if progress and (completed in (256, 512, 768) or completed % 2048 == 0):
                progress(completed, len(rows))
    return scores


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_file(path, digest):
    if sha(Path(path)) != digest:
        raise ValueError(f"SHA-256 mismatch: {path}")


def verify_inputs(config, reference):
    if reference.get("status") != "PASS_NATIVE_FIT_TRAIN" or reference.get("variant") != "baseline":
        raise ValueError("completed baseline native fit receipt required")
    if sha(CONFIG) != reference.get("config_sha256"):
        raise ValueError("effective fit config differs from the fitted run")

    pins = reference["validated_hashes"]
    corpus = Path(config["corpus"])
    queries = Path(config["queries"])
    initial_readout = Path(config["initial_readout"])
    required_paths = [
        (corpus / "result.json", pins[str(corpus / "result.json")]),
        (queries / "result.json", pins[str(queries / "result.json")]),
        (initial_readout, pins[str(initial_readout)]),
        (CONFIG, reference["config_sha256"]),
    ]
    for path, digest in required_paths:
        verify_file(path, digest)

    for root in (corpus, queries):
        receipt = load_json(root / "result.json")
        for name, digest in receipt["artifacts"].items():
            verify_file(root / name, digest)

    source_paths = [
        "scripts/train_native_common.py",
        "scripts/native_fit.py",
        "scripts/native_fit_runtime.py",
        "scripts/native_candidate_inputs.py",
        "src/tiny_omni_decision/native_candidate.py",
    ]
    source_commit = reference["source_commit"]
    for name in source_paths:
        original = str(SOURCE_ROOT / name)
        if original not in pins:
            raise ValueError(f"training source pin missing: {name}")
        verify_file(SOURCE_ROOT / name, pins[original])
        trained_blob = subprocess.check_output(
            ["git", "rev-parse", f"{source_commit}:{name}"], cwd=REPO, text=True
        ).strip()
        current_blob = subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{name}"], cwd=REPO, text=True
        ).strip()
        if trained_blob != current_blob:
            raise ValueError(f"current runtime source differs from fit source: {name}")
    return corpus, queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha(args.reference / "result.json") != args.reference_sha256:
        raise ValueError("reference receipt hash differs")
    reference = load_json(args.reference / "result.json")
    config = load_json(CONFIG)
    corpus, queries = verify_inputs(config, reference)
    if reference.get("best_step") != 1792 or reference.get("validation_count") != 24346:
        raise ValueError("unexpected pinned baseline run metadata")

    best_path = args.reference / "best.pt"
    expected_step = reference["best_step"]
    verify_file(best_path, reference["artifacts"]["best.pt"])
    expected_scores_path = args.reference / f"validation-{expected_step:04}.npy"
    verify_file(expected_scores_path, reference["artifacts"][expected_scores_path.name])

    rows = [
        json.loads(line)
        for line in (corpus / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != reference["validation_count"] or any(
        row.get("split") != "validation" for row in rows
    ):
        raise ValueError("validation corpus differs from the pinned split")
    expected_ids = load_json(args.reference / "validation-ids.json")
    ids = [
        dict(id=row["id"], option_ids=row["option_ids"], target_index=row["target_index"])
        for row in rows
    ]
    if ids != expected_ids:
        raise ValueError("validation IDs/order/options/targets differ from fitted run")

    observations = load_json(corpus / "observations.json")
    banks = load_json(corpus / "banks.json")
    query_receipt = load_json(queries / "result.json")
    banks["new/queries.npy"] = dict(
        path=str(queries / "queries.npy"),
        sha256=query_receipt["artifacts"]["queries.npy"],
        shape=[2622, 384],
    )
    query_map = load_json(corpus / "query-map.json")

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    if checkpoint.get("step") != expected_step:
        raise ValueError("best checkpoint step differs from fit receipt")
    model = NativeCandidateReadout()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.cpu()
    store = FeatureStore(observations, banks, query_map, device="cpu")

    args.output.mkdir(parents=True, exist_ok=False)
    exported = args.output / "readout.pt"
    torch.save(model.state_dict(), exported)
    reloaded = NativeCandidateReadout()
    exported_state = torch.load(exported, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(exported_state, strict=True)
    reloaded.eval()

    started = time.perf_counter()
    last_tick = started
    last_count = 0
    throughput_samples = []

    def progress(completed, total):
        nonlocal last_tick, last_count
        now = time.perf_counter()
        interval = now - last_tick
        rate = (completed - last_count) / interval if interval else math.inf
        throughput_samples.append(dict(examples=completed, interval_examples=completed - last_count,
                                       interval_seconds=interval, examples_per_second=rate))
        average_rate = float(
            np.mean([sample["examples_per_second"] for sample in throughput_samples[-3:]])
        )
        remaining = (total - completed) / average_rate if average_rate else math.inf
        print(
            f"validation {completed}/{total}; sample {rate:.1f} examples/s; "
            f"last-three mean {average_rate:.1f} examples/s; "
            f"estimated remaining {remaining / 60:.1f} min",
            flush=True,
        )
        last_tick, last_count = now, completed

    scores = score_validation(reloaded, rows, store, progress=progress)
    elapsed = time.perf_counter() - started
    expected_scores = np.load(expected_scores_path, allow_pickle=False)
    if expected_scores.shape != scores.shape:
        raise ValueError("saved validation score matrix shape differs")
    max_abs = float(np.max(np.abs(scores - expected_scores)))
    if max_abs > SCORE_ATOL:
        raise ValueError(f"CPU score drift {max_abs} exceeds absolute tolerance {SCORE_ATOL}")
    option_disagreements = int(
        np.count_nonzero(scores.argmax(axis=1) != expected_scores.argmax(axis=1))
    )

    score_path = args.output / "validation-scores.npy"
    np.save(score_path, scores, allow_pickle=False)
    metrics = summarize(rows, scores)
    reference_metrics = summarize(rows, expected_scores)
    metric_deltas = {}
    for group in ("overall", "macro", "modalities"):
        metric_deltas[group] = {}
        for name, values in metrics[group].items():
            if isinstance(values, dict):
                metric_deltas[group][name] = {
                    key: values[key] - reference_metrics[group][name][key]
                    for key in values
                    if key in reference_metrics[group][name]
                    and isinstance(values[key], (float, int))
                    and isinstance(reference_metrics[group][name][key], (float, int))
                }
            elif isinstance(values, (float, int)):
                metric_deltas[group][name] = values - reference_metrics[group][name]
    metrics_path = args.output / "validation-metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifacts = {
        path.name: sha(path)
        for path in (exported, score_path, metrics_path)
    }
    report = dict(
        status="PASS_CPU_CACHED_FEATURE_REPLAY",
        kind="full_validation_readout_runtime_parity",
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        reference_receipt_sha256=args.reference_sha256,
        reference_checkpoint_sha256=sha(best_path),
        selected_step=expected_step,
        validation_count=len(rows),
        validation_ids_order_sha256=hashlib.sha256(
            json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
        score_absolute_tolerance=SCORE_ATOL,
        max_abs_score_difference=max_abs,
        top_option_disagreements=option_disagreements,
        reference_metrics=reference_metrics,
        metric_deltas_vs_saved_gpu_scores=metric_deltas,
        cpu_only=True,
        cuda_used=False,
        training_performed=False,
        checkpoint_selection_performed=False,
        sealed_audit_loaded=False,
        raw_encoder_inference=False,
        validation_is_exposed_development_data=True,
        elapsed_seconds=elapsed,
        examples_per_second=len(rows) / elapsed,
        throughput_samples=throughput_samples,
        artifact_sha256=artifacts,
        note=(
            "CPU parity over cached features only; not raw encoder inference or an independent "
            "quality estimate."
        ),
    )
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
