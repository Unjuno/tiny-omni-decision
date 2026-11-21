"""Frozen-checkpoint input-use diagnostic on the already exposed R2 validation.

This is a mechanism check, not a new benchmark or checkpoint-selection run.
It loads only the selected baseline readout and existing development inputs;
the sealed audit is never referenced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path(r"C:\CodexWorktrees\pretrained-reuse-omni")
DEFAULT_BASELINE = Path(r"C:\CodexArtifacts\pretrained-reuse-r2\native-common-baseline-fit-01")
DEFAULT_CORPUS = Path(r"C:\CodexArtifacts\pretrained-reuse-r2\native-common-corpus-02")
DEFAULT_QUERIES = Path(r"C:\CodexArtifacts\pretrained-reuse-r2\native-common-queries-cache-01")
DEFAULT_OUTPUT = Path(r"C:\CodexArtifacts\pretrained-reuse-r2\native-weak-input-use-01")
EXPECTED_BASELINE_RECEIPT = "aaa1fe96a4be85a71ce146981c6b2e8b58e1308d91fc110e1ece6804b4d8048c"
EXPECTED_BASELINE_CONFIG = "45954a7240ee8da32e4b5b5dc3283994c40c4849451f27ff4fce13d3cc118f75"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_donors(rows: list[dict], profiles: dict[str, tuple] | None = None) -> dict[str, dict]:
    """Map each row to a different-group donor within the same task/profile."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        profile = profiles.get(row["observation"], ()) if profiles else ()
        signature = (row["modality"], row["source"], row["task"], len(row["options"]), profile)
        buckets[signature].append(row)
    donors = {}
    for bucket in buckets.values():
        ordered = sorted(bucket, key=lambda row: (row["group"], row["id"]))
        if len({row["group"] for row in ordered}) < 2:
            continue
        for i, row in enumerate(ordered):
            for offset in range(1, len(ordered)):
                candidate = ordered[(i + offset) % len(ordered)]
                if candidate["group"] != row["group"]:
                    donors[row["id"]] = candidate
                    break
    return donors


def zero_blocks(blocks):
    """Zero only native feature content; preserve token type, positions and masks."""
    from tiny_omni_decision.native_candidate import NativeTokenBlock

    return [
        NativeTokenBlock(
            block.kind, block.values.new_zeros(block.values.shape), block.positions, block.valid
        )
        for block in blocks
    ]


def replace_block_values(blocks, donor_blocks):
    """Use a donor's feature values while retaining the recipient coordinates."""
    from tiny_omni_decision.native_candidate import NativeTokenBlock

    if len(blocks) != len(donor_blocks):
        raise ValueError("donor feature-block count differs")
    result = []
    for current, donor in zip(blocks, donor_blocks, strict=True):
        if current.kind != donor.kind or current.values.shape != donor.values.shape:
            raise ValueError("donor feature profile differs")
        result.append(
            NativeTokenBlock(current.kind, donor.values, current.positions, current.valid)
        )
    return result


def row_metrics(rows: list[dict], scores: np.ndarray) -> dict[str, np.ndarray]:
    """Per-row paired metrics for group-cluster uncertainty intervals."""
    correct, nll, brier, confidence = [], [], [], []
    for row, raw in zip(rows, scores, strict=True):
        count = len(row["options"])
        logits = raw[:count].astype(np.float64)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        target = row["target_index"]
        truth = np.zeros(count, dtype=np.float64)
        truth[target] = 1.0
        correct.append(float(int(probs.argmax()) == target))
        nll.append(float(-math.log(max(probs[target], 1e-300))))
        brier.append(float(np.square(probs - truth).sum()))
        confidence.append(float(probs.max()))
    return {
        "accuracy": np.asarray(correct),
        "nll": np.asarray(nll),
        "brier": np.asarray(brier),
        "confidence": np.asarray(confidence),
    }


def group_bootstrap_deltas(rows, baseline_scores, variant_scores, *, seed=17, draws=5000):
    """Paired row-weighted deltas with whole-group percentile bootstrap CIs."""
    groups = sorted({row["group"] for row in rows})
    positions = {group: i for i, group in enumerate(groups)}
    per_row = np.asarray([positions[row["group"]] for row in rows], dtype=np.int64)
    grouped = np.bincount(per_row, minlength=len(groups))
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(groups), size=(draws, len(groups)))
    denominator = grouped[resamples].sum(axis=1)
    before = row_metrics(rows, baseline_scores)
    after = row_metrics(rows, variant_scores)
    results = {}
    for metric in ("accuracy", "nll", "brier"):
        b = np.bincount(per_row, weights=before[metric], minlength=len(groups))
        a = np.bincount(per_row, weights=after[metric], minlength=len(groups))
        delta = (a[resamples].sum(axis=1) - b[resamples].sum(axis=1)) / denominator
        results[metric] = {
            "point_delta": float(after[metric].mean() - before[metric].mean()),
            "ci95_percentile": [float(v) for v in np.quantile(delta, [0.025, 0.975])],
        }
    return results


def _receipt_path_hash(receipt: dict, relative_path: str) -> str:
    suffix = "\\" + relative_path.replace("/", "\\").lower()
    matches = [
        value
        for path, value in receipt["validated_hashes"].items()
        if path.lower().endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError("unique pinned source hash required for " + relative_path)
    return matches[0]


def verify_pinned_source(source_root: Path, baseline: dict) -> dict:
    required = (
        "scripts/native_fit_runtime.py",
        "scripts/native_candidate_inputs.py",
        "scripts/native_fit.py",
        "scripts/semantic_state_diagnostic.py",
        "src/tiny_omni_decision/native_candidate.py",
    )
    verified = {}
    for relative in required:
        path = source_root / relative
        actual = sha256(path)
        expected = _receipt_path_hash(baseline, relative)
        if actual != expected:
            raise ValueError("pinned inference source differs: " + relative)
        verified[str(path)] = actual
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError("diagnostic output must be a new directory")
    if args.bootstrap_draws < 100:
        raise ValueError("at least 100 bootstrap draws required")
    usage = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    memory, utilization = [int(value.strip()) for value in usage.split(",")]
    if memory > 512 or utilization > 5:
        raise RuntimeError("GPU occupied; do not compete")

    baseline_path = args.baseline / "result.json"
    baseline_sha = sha256(baseline_path)
    if baseline_sha != EXPECTED_BASELINE_RECEIPT:
        raise ValueError("selected baseline receipt hash changed")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if (
        baseline.get("status") != "PASS_NATIVE_FIT_TRAIN"
        or baseline.get("variant") != "baseline"
        or baseline.get("best_step") != 1792
        or baseline.get("sealed_audit_loaded") is not False
        or baseline.get("config_sha256") != EXPECTED_BASELINE_CONFIG
    ):
        raise ValueError("expected fixed selected baseline is required")
    source_hashes = verify_pinned_source(args.source_root, baseline)

    # Import the exact inference implementation whose files the training receipt pinned.
    sys.path.insert(0, str(args.source_root))
    sys.path.insert(0, str(args.source_root / "src"))
    import torch

    from scripts.native_fit_runtime import FeatureStore
    from tiny_omni_decision.native_candidate import NativeCandidateReadout

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered input-use check")
    device = "cuda"

    config_path = args.source_root / "configs/pretrained_reuse/native_common_fit_v1.json"
    if sha256(config_path) != EXPECTED_BASELINE_CONFIG:
        raise ValueError("pinned baseline config changed")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for path, digest in config["input_pins"].items():
        if sha256(Path(path)) != digest:
            raise ValueError("pinned config input changed: " + path)
    corpus_receipt = json.loads((args.corpus / "result.json").read_text(encoding="utf-8"))
    query_receipt = json.loads((args.queries / "result.json").read_text(encoding="utf-8"))
    if (
        sha256(args.corpus / "result.json")
        != config["input_pins"][str(args.corpus / "result.json")]
    ):
        raise ValueError("pinned corpus receipt differs")
    if (
        sha256(args.queries / "result.json")
        != config["input_pins"][str(args.queries / "result.json")]
    ):
        raise ValueError("pinned query receipt differs")
    for name, digest in corpus_receipt["artifacts"].items():
        if sha256(args.corpus / name) != digest:
            raise ValueError("pinned corpus artifact differs: " + name)
    for name, digest in query_receipt["artifacts"].items():
        if sha256(args.queries / name) != digest:
            raise ValueError("pinned query artifact differs: " + name)

    validation = [
        json.loads(line)
        for line in (args.corpus / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(validation) != 24346 or any(row["split"] != "validation" for row in validation):
        raise ValueError("fixed exposed development validation corpus required")
    selected = [row for row in validation if row["modality"] in ("image", "text")]
    counts = {
        modality: sum(row["modality"] == modality for row in selected)
        for modality in ("image", "text")
    }
    if counts != {"image": 2048, "text": 10732}:
        raise ValueError("fixed Image/Text validation counts differ")

    banks = json.loads((args.corpus / "banks.json").read_text(encoding="utf-8"))
    banks["new/queries.npy"] = {
        "path": str(args.queries / "queries.npy"),
        "sha256": query_receipt["artifacts"]["queries.npy"],
        "shape": [2622, 384],
    }
    observations = json.loads((args.corpus / "observations.json").read_text(encoding="utf-8"))
    query_map = json.loads((args.corpus / "query-map.json").read_text(encoding="utf-8"))
    used_bank_names = {
        feature["bank"]
        for row in selected
        for feature in observations[row["observation"]]["features"]
    } | {"new/queries.npy"}
    relevant_bank_hashes = {}
    for name in sorted(used_bank_names):
        metadata = banks[name]
        path = Path(metadata["path"])
        actual = sha256(path)
        if actual != metadata["sha256"]:
            raise ValueError("used validation feature bank hash differs: " + str(path))
        relevant_bank_hashes[str(path)] = actual
    profiles = {
        ident: tuple((feature["kind"], feature["tokens"]) for feature in value["features"])
        for ident, value in observations.items()
    }

    model = NativeCandidateReadout()
    initial_path = Path(config["initial_readout"])
    if sha256(initial_path) != config["input_pins"][str(initial_path)]:
        raise ValueError("pinned initial readout differs")
    model.load_state_dict(
        torch.load(initial_path, map_location="cpu", weights_only=True), strict=True
    )
    best_path = args.baseline / "best.pt"
    expected_best_sha = baseline["artifacts"]["best.pt"]
    if sha256(best_path) != expected_best_sha:
        raise ValueError("selected baseline checkpoint hash differs")
    saved = torch.load(best_path, map_location="cpu", weights_only=True)
    if saved["step"] != 1792:
        raise ValueError("baseline checkpoint step differs")
    model.load_state_dict(saved["model"], strict=True)
    model = model.to(device).eval()
    store = FeatureStore(observations, banks, query_map, device=device)
    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    (output_root / "run-state.json").write_text(
        json.dumps(
            {"status": "RUNNING", "pid": os.getpid(), "started_utc": datetime.now(UTC).isoformat()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    expected_all = np.load(args.baseline / "validation-1792.npy", allow_pickle=False)
    if (
        sha256(args.baseline / "validation-1792.npy")
        != baseline["artifacts"]["validation-1792.npy"]
    ):
        raise ValueError("saved selected validation predictions changed")
    full_index = {row["id"]: i for i, row in enumerate(validation)}

    donor_by_modality = {
        modality: choose_donors([row for row in selected if row["modality"] == modality], profiles)
        for modality in ("image", "text")
    }
    for modality in ("image", "text"):
        subset = [row for row in selected if row["modality"] == modality]
        donors = donor_by_modality[modality]
        if len(donors) != len(subset):
            raise ValueError("same-task, different-group donors not available for every row")
        conditions = {
            name: []
            for name in (
                "baseline",
                "observation_zero",
                "observation_shuffle",
                "candidate_query_zero",
            )
        }
        with torch.inference_mode():
            for row in subset:
                blocks, queries = store.get(row)
                baseline_scores = model(blocks, queries).float().cpu().numpy()
                expected = expected_all[full_index[row["id"]], : len(row["options"])]
                if not np.array_equal(baseline_scores, expected):
                    raise ValueError(
                        "pinned baseline checkpoint replay differs from saved validation"
                    )
                donor_blocks, _ = store.get(donors[row["id"]])
                variants = {
                    "baseline": (blocks, queries),
                    "observation_zero": (zero_blocks(blocks), queries),
                    "observation_shuffle": (replace_block_values(blocks, donor_blocks), queries),
                    "candidate_query_zero": (blocks, torch.zeros_like(queries)),
                }
                for name, (features, query) in variants.items():
                    values = (
                        baseline_scores
                        if name == "baseline"
                        else model(features, query).float().cpu().numpy()
                    )
                    padded = np.full(40, -1e30, dtype=np.float32)
                    padded[: len(values)] = values
                    conditions[name].append(padded)
        for name, values in conditions.items():
            conditions[name] = np.stack(values)
            np.save(output_root / f"{modality}-{name}.npy", conditions[name], allow_pickle=False)

        from scripts.native_fit import summarize

        summaries = {
            name: summarize(subset, scores)["modality_macro"][modality]
            for name, scores in conditions.items()
        }
        intervals = {
            name: group_bootstrap_deltas(
                subset,
                conditions["baseline"],
                conditions[name],
                seed=17,
                draws=args.bootstrap_draws,
            )
            for name in conditions
            if name != "baseline"
        }
        donor_manifest = [
            {
                "row_id": row["id"],
                "donor_id": donors[row["id"]]["id"],
                "same_source_task": True,
                "different_group": True,
            }
            for row in subset
        ]
        (output_root / f"{modality}-donors.json").write_text(
            json.dumps(donor_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output_root / f"{modality}-metrics.json").write_text(
            json.dumps(
                {
                    "count": len(subset),
                    "groups": len({r["group"] for r in subset}),
                    "metrics": summaries,
                    "paired_group_bootstrap": intervals,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    result = {
        "status": "PASS_NATIVE_WEAK_INPUT_USE_DIAGNOSTIC",
        "phase": "frozen_selected_checkpoint_input_interventions",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "script_sha256": sha256(Path(__file__)),
        "baseline_receipt_sha256": baseline_sha,
        "baseline_checkpoint_sha256": expected_best_sha,
        "baseline_selected_step": 1792,
        "inference_source_hashes": source_hashes,
        "corpus_receipt_sha256": sha256(args.corpus / "result.json"),
        "validation_sha256": corpus_receipt["artifacts"]["validation.jsonl"],
        "query_receipt_sha256": sha256(args.queries / "result.json"),
        "used_validation_feature_bank_hashes": relevant_bank_hashes,
        "validation_rows_loaded": len(validation),
        "scored_counts": counts,
        "selected_validation_exposed": True,
        "sealed_audit_loaded": False,
        "training_performed": False,
        "optimizer_updates": 0,
        "calibration_fitted": False,
        "interventions": [
            "observation content zeroed",
            "same-task/different-group observation shuffle",
            "candidate query vectors zeroed",
        ],
        "bootstrap": {
            "resampling_unit": "whole corpus group within modality",
            "draws": args.bootstrap_draws,
            "seed": 17,
            "interval": "paired percentile 95%",
            "ece_interval": False,
        },
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tf32": False,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "wall_seconds": time.perf_counter() - start,
        "baseline_saved_logit_replay_max_abs": 0.0,
        "cloud_cost": 0,
        "gpu_preflight": usage,
        "finished_utc": datetime.now(UTC).isoformat(),
    }
    result["artifacts"] = {
        path.name: sha256(path)
        for path in sorted(output_root.iterdir())
        if path.is_file() and path.name not in {"result.json", "run-state.json"}
    }
    (output_root / "run-state.json").write_text(
        json.dumps({"status": "COMPLETE", "finished_utc": result["finished_utc"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
