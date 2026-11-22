"""Verify saved TRAIN-only TallyQA group-CV outputs without fitting models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.cv_tallyqa_vg_train_groups import (
    IMAGE_WIDTH,
    TEXT_WIDTH,
    _read_train_image_features,
    assign_group_folds,
    group_keys,
)
from scripts.train_tallyqa_vg_readout import (
    _batch_logits,
    _encode_rows,
    _metrics,
    _pooled_texts,
)
from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_artifact_index(run_dir: Path) -> dict[str, Any]:
    index = _read_json(run_dir / "artifact-index.json")
    entries = index.get("artifacts")
    if not isinstance(entries, list) or len(entries) != index.get("artifact_count"):
        raise ValueError("artifact index count is inconsistent")
    seen: set[str] = set()
    total = 0
    for entry in entries:
        name = entry.get("path")
        if not isinstance(name, str) or Path(name).name != name or name in seen:
            raise ValueError("artifact index contains an invalid or duplicate path")
        seen.add(name)
        path = run_dir / name
        if not path.is_file():
            raise ValueError(f"indexed artifact is missing: {name}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != entry.get("bytes") or digest != entry.get("sha256"):
            raise ValueError(f"indexed artifact hash/size mismatch: {name}")
        total += size
    actual = {
        path.name
        for path in run_dir.iterdir()
        if path.is_file() and path.name != "artifact-index.json"
    }
    if seen != actual or total != index.get("total_bytes"):
        raise ValueError("artifact index does not cover the exact run contents")
    return {"artifact_count": len(seen), "total_bytes": total,
            "artifact_index_sha256": sha256_file(run_dir / "artifact-index.json")}


def verify_oof_identity(
    rows: list[dict[str, Any]], oof_rows: list[dict[str, Any]], *, n_folds: int = 5
) -> dict[str, Any]:
    if len(rows) != len(oof_rows):
        raise ValueError("OOF row count differs from TRAIN-only manifest")
    keys = group_keys(rows)
    assignment = assign_group_folds(rows, keys, n_folds=n_folds, seed=17)
    seen: set[str] = set()
    for source, record, group in zip(rows, oof_rows, keys, strict=True):
        qid = str(source["question_id"])
        if qid in seen or str(record.get("question_id")) != qid:
            raise ValueError("OOF question IDs are duplicated or out of source order")
        seen.add(qid)
        expected_target = int(source["target"])
        if (
            str(record.get("image_id")) != str(source["image_id"])
            or int(record.get("target_option_index", -1)) != expected_target
            or int(record.get("fold", -1)) != assignment[group]
            or record.get("group_key") != group
        ):
            raise ValueError(f"OOF identity/split/target mismatch for {qid}")
        for arm in ("joint", "question_only", "image_zero"):
            pred = record.get("arms", {}).get(arm)
            if not isinstance(pred, dict) or str(pred.get("question_id")) != qid:
                raise ValueError(f"missing OOF prediction for {qid}/{arm}")
            probabilities = np.asarray(pred.get("probabilities"), dtype=np.float64)
            if (
                probabilities.shape != (4,)
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or not np.isclose(probabilities.sum(), 1.0, atol=1e-6)
            ):
                raise ValueError(f"invalid OOF probabilities for {qid}/{arm}")
            options = [str(option["id"]) for option in source["options"]]
            if pred.get("options") != options:
                raise ValueError(f"OOF option order differs for {qid}/{arm}")
            pred_index = int(np.argmax(probabilities))
            if (
                pred.get("prediction") != options[pred_index]
                or pred.get("target") != options[expected_target]
            ):
                raise ValueError(f"OOF label does not match probabilities for {qid}/{arm}")
    return {"train_rows": len(rows), "unique_question_ids": len(seen),
            "outer_folds": n_folds, "ordered_question_ids_sha256": hashlib.sha256(
                "\n".join(str(row["question_id"]) for row in rows).encode("utf-8")
            ).hexdigest()}


def _aggregate_saved_metrics(
    oof_rows: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    targets = [int(row["target"]) for row in rows]
    result = {}
    for arm in ("joint", "question_only", "image_zero"):
        probabilities = torch.tensor([
            record["arms"][arm]["probabilities"] for record in oof_rows
        ], dtype=torch.float64)
        logits = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        result[arm] = _metrics([logits[index] for index in range(len(rows))], targets)
    return result


def _replay(
    rows: list[dict[str, Any]],
    tasks: list[torch.Tensor],
    oof_rows: list[dict[str, Any]],
    run_dir: Path,
    *,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, Any]:
    keys = group_keys(rows)
    assignment = assign_group_folds(rows, keys, n_folds=5, seed=17)
    deltas = {name: [] for name in ("joint", "question_only", "image_zero")}
    max_delta = {name: 0.0 for name in deltas}
    prediction_flips = {name: 0 for name in deltas}
    counts = {name: 0 for name in deltas}
    for fold in range(5):
        indices = [i for i, group in enumerate(keys) if assignment[group] == fold]
        for arm, feature_size in (("joint", IMAGE_WIDTH + 2 * TEXT_WIDTH),
                                  ("question_only", 2 * TEXT_WIDTH)):
            model = FrozenFeatureCandidateScorer(feature_size, hidden_size=128).to(device)
            checkpoint = run_dir / f"fold-{fold:02d}-{arm}-selected.pt"
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            model.eval()
            score_tasks = [
                tasks[i] if arm == "joint" else tasks[i][:, IMAGE_WIDTH:]
                for i in indices
            ]
            with torch.inference_mode():
                logits = torch.cat([
                    _batch_logits(model, score_tasks, device,
                                  list(range(start, min(start + batch_size, len(score_tasks)))))
                    for start in range(0, len(score_tasks), batch_size)
                ])
                if arm == "joint":
                    zero_tasks = [value.clone() for value in score_tasks]
                    for value in zero_tasks:
                        value[:, :IMAGE_WIDTH] = 0
                    zero_logits = torch.cat([
                        _batch_logits(model, zero_tasks, device,
                                      list(range(start, min(start + batch_size, len(zero_tasks)))))
                        for start in range(0, len(zero_tasks), batch_size)
                    ])
            predicted = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            references = np.asarray([
                oof_rows[i]["arms"][arm]["probabilities"] for i in indices
            ], dtype=np.float64)
            delta = np.abs(predicted - references)
            deltas[arm].extend(delta.ravel().tolist())
            max_delta[arm] = max(max_delta[arm], float(delta.max(initial=0.0)))
            prediction_flips[arm] += int((predicted.argmax(1) != references.argmax(1)).sum())
            counts[arm] += len(indices)
            if arm == "joint":
                predicted_zero = torch.softmax(zero_logits.float(), dim=-1).cpu().numpy()
                references_zero = np.asarray([
                    oof_rows[i]["arms"]["image_zero"]["probabilities"] for i in indices
                ], dtype=np.float64)
                delta_zero = np.abs(predicted_zero - references_zero)
                deltas["image_zero"].extend(delta_zero.ravel().tolist())
                max_delta["image_zero"] = float(delta_zero.max(initial=0.0))
                prediction_flips["image_zero"] = int(
                    (predicted_zero.argmax(1) != references_zero.argmax(1)).sum()
                )
                counts["image_zero"] += len(indices)
            del model
    return {
        "method": "load selected fold checkpoints and replay exact outer TRAIN rows; no fitting",
        "count_by_arm": counts,
        "max_absolute_probability_delta": max_delta,
        "argmax_flips": prediction_flips,
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve(strict=True)
    manifest_path = args.task_manifest.resolve(strict=True)
    task = _read_json(manifest_path)
    rows = task.get("train")
    if task.get("split_scope") != "train_only" or any(
        key in task for key in ("validation", "test", "sealed_audit")
    ) or not isinstance(rows, list) or not rows:
        raise ValueError("verifier requires a train-only manifest without held-out partitions")
    result = _read_json(run_dir / "result.json")
    if result.get("status") != "complete_train_only_nested_group_cv":
        raise ValueError("run is not marked complete")
    if (
        result.get("official_validation_loaded")
        or result.get("official_test_loaded")
        or result.get("sealed_audit_loaded")
    ):
        raise ValueError("run metadata indicates held-out data were loaded")
    indexed = verify_artifact_index(run_dir)
    oof_rows = [json.loads(line) for line in (run_dir / "oof-predictions.jsonl").read_text(
        encoding="utf-8").splitlines() if line]
    identity = verify_oof_identity(rows, oof_rows)
    metrics = _aggregate_saved_metrics(oof_rows, rows)
    metric_keys = {
        "joint": "joint_image_question",
        "question_only": "question_only",
        "image_zero": "joint_with_image_zeroed",
    }
    metric_delta = {}
    for arm, result_key in metric_keys.items():
        metric_delta[arm] = {}
        for metric_name in ("accuracy", "nll", "brier", "ece_15_bins"):
            difference = abs(
                float(metrics[arm][metric_name]) - float(result["metrics"][result_key][metric_name])
            )
            metric_delta[arm][metric_name] = difference
            if difference > 2e-6:
                raise ValueError(
                    "saved OOF metric disagrees with result.json: "
                    f"{arm}/{metric_name}"
                )
    cache_report = _read_json(args.cache_report.resolve(strict=True))
    cache_path = args.feature_cache.resolve(strict=True)
    model_root = args.text_model.resolve(strict=True)
    if (
        sha256_file(manifest_path) != result.get("train_only_manifest_sha256")
        or sha256_file(cache_path) != result.get("feature_cache_sha256")
        or sha256_file(model_root / "model.safetensors") != result.get("text_model_sha256")
        or cache_report.get("cache_sha256") != sha256_file(cache_path)
        or cache_report.get("task_manifest_sha256") != task.get("source_task_manifest_sha256")
    ):
        raise ValueError("frozen input hash check failed")
    image_features = _read_train_image_features(cache_path, rows)
    unique_texts: dict[str, str] = {}
    for row in rows:
        qid = str(row["question_id"])
        unique_texts[qid] = str(row["question"])
        for index, option in enumerate(row["options"]):
            unique_texts[f"option:{qid}:{index}"] = f"Candidate count: {option['text']}."
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
    embeddings = _pooled_texts(list(unique_texts.values()), model_root, device)
    tasks, _ = _encode_rows(rows, image_features, dict(zip(unique_texts, embeddings, strict=True)))
    replay = _replay(rows, tasks, oof_rows, run_dir, device=device)
    probability_pass = all(
        value <= 1e-5 for value in replay["max_absolute_probability_delta"].values()
    )
    return {
        "status": (
            "verified"
            if probability_pass and all(value == 0 for value in replay["argmax_flips"].values())
            else "replay_mismatch"
        ),
        "source_run_result_sha256": sha256_file(run_dir / "result.json"),
        "artifact_integrity": indexed,
        "oof_integrity": identity,
        "metrics_recomputed_from_saved_probabilities": metrics,
        "metric_delta_from_result_json": metric_delta,
        "checkpoint_replay": replay,
        "replay_probability_tolerance": 1e-5,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "sealed_audit_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite verifier report: {output}")
    report = verify(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
