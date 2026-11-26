"""One-shot comparison of frozen Clotho question-only and audio-question readouts."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from scripts.starss23_readout_common import binary_metrics
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
ATTEMPT = ROOT / "clotho-aqa-publisher-val-20261126-attempt-01"
OUTPUT = ATTEMPT / "result"
CHECKPOINTS = {
    "question_only": ROOT / "clotho-aqa-question-control-01",
    "audio_question": ROOT / "clotho-aqa-joint-control-01",
}
EXPECTED_CHECKPOINT_SHA256 = {
    "question_only": "fa98ddcebc3c0175d17794ccaabb820b03d5fc600740cb3c4deaa77419af4df3",
    "audio_question": "ac3fc2012c146538537a293e007a386fd1c738ba6ee0a7452cc89ccf05daab4e",
}
EXPECTED_NORMALIZATION_SHA256 = {
    "question_only": "9c7b39b9945baaa091aa39f512b3a68d099ed0d68237bc2e89eba04df7a96850",
    "audio_question": "f152042f1f1ff03a4affb3381b5419dfe390edf64420dbd1ed178e835bbe1791",
}
EXPECTED_MANIFEST_SHA256 = "100630d59a92f461760a83aac1bfd6684d511904a42cf6ea1c71819b0fe5fb19"
EXPECTED_INTERNAL_VALIDATION_IDS_SHA256 = (
    "273197786d3eba4b88b8394d5ac807e08243ad74f1912858123ee7a6db2868a6"
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 17


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_cluster_bootstrap(
    targets: np.ndarray,
    probabilities: dict[str, np.ndarray],
    clusters: list[str],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Bootstrap paired metric deltas by uploader, retaining all questions per draw."""
    if len(targets) != len(clusters) or any(
        len(p) != len(targets) for p in probabilities.values()
    ):
        raise ValueError("bootstrap inputs must have matching row counts")
    if len(probabilities) != 2:
        raise ValueError("paired bootstrap requires exactly two conditions")
    unique_clusters = sorted(set(clusters))
    if len(unique_clusters) < 2:
        raise ValueError("at least two independent uploader clusters are required")
    indices = {
        cluster: np.flatnonzero(np.asarray(clusters) == cluster)
        for cluster in unique_clusters
    }
    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {"accuracy": [], "nll": [], "brier": [], "ece": []}
    names = list(probabilities)
    for _ in range(replicates):
        draw = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        sample = np.concatenate([indices[cluster] for cluster in draw])
        values = {
            name: binary_metrics(targets[sample], probabilities[name][sample])
            for name in names
        }
        for metric in deltas:
            deltas[metric].append(values[names[1]][metric] - values[names[0]][metric])
    return {
        "cluster_count": len(unique_clusters),
        "replicates": replicates,
        "seed": seed,
        "delta_order": f"{names[1]} minus {names[0]}",
        "interval_method": "paired uploader-cluster percentile 95% interval",
        "deltas": {
            metric: {
                "point": float(
                    binary_metrics(targets, probabilities[names[1]])[metric]
                    - binary_metrics(targets, probabilities[names[0]])[metric]
                ),
                "ci95": [float(x) for x in np.quantile(values, [0.025, 0.975])],
            }
            for metric, values in deltas.items()
        },
    }


def build_features(audio: np.ndarray, question: np.ndarray, condition: str) -> torch.Tensor:
    if audio.shape != question.shape or audio.ndim != 2 or audio.shape[1] != 384:
        raise ValueError("aligned 384-dimensional feature matrices required")
    if condition == "question_only":
        audio = np.zeros_like(audio)
    elif condition != "audio_question":
        raise ValueError(f"unknown frozen condition: {condition}")
    shared = np.concatenate((audio, question), axis=1)
    options = np.repeat(shared[:, None, :], 2, axis=1)
    ids = np.broadcast_to(np.eye(2, dtype=np.float32), (len(shared), 2, 2))
    return torch.from_numpy(np.concatenate((options, ids), axis=-1).astype(np.float32))


def main() -> dict:
    started = time.perf_counter()
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    manifest_path = ATTEMPT / "inclusion-manifest.jsonl"
    if sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("cohort inclusion manifest hash differs from pre-registration")
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    if len(manifest) != 244:
        raise ValueError("frozen cohort row count changed")
    if any(
        row["license"] != "http://creativecommons.org/publicdomain/zero/1.0/"
        or len(row["votes"]) != 3
        or any(vote not in {"yes", "no"} for vote in row["votes"])
        or row["target"] != int(row["votes"].count("yes") > row["votes"].count("no"))
        for row in manifest
    ):
        raise ValueError("cohort differs from the exact-CC0 three-vote binary protocol")
    features_dir = ATTEMPT / "features"
    feature_report = json.loads((features_dir / "result.json").read_text(encoding="utf-8"))
    if feature_report.get("status") != "PASS":
        raise ValueError("frozen feature extraction has not passed provenance/parity checks")
    questions = np.load(features_dir / "question-features.npy", allow_pickle=False)
    question_ids = json.loads((features_dir / "question-ids.json").read_text(encoding="utf-8"))
    audio = np.load(features_dir / "audio-features.npy", allow_pickle=False)
    audio_ids = json.loads((features_dir / "audio-ids.json").read_text(encoding="utf-8"))
    expected_questions = [[row["source_id"], row["normalized_question"]] for row in manifest]
    if question_ids != expected_questions or questions.shape != (244, 384):
        raise ValueError("question features are not in frozen cohort order")
    if audio_ids != sorted({row["source_id"] for row in manifest}):
        raise ValueError("audio feature order differs from frozen cohort")
    if audio.shape != (len(audio_ids), 384) or not np.isfinite(audio).all():
        raise ValueError("invalid audio features")
    audio_lookup = {source_id: i for i, source_id in enumerate(audio_ids)}
    question_audio = audio[[audio_lookup[row["source_id"]] for row in manifest]]
    targets = np.asarray([row["target"] for row in manifest], dtype=np.int64)
    conditions: dict[str, np.ndarray] = {}
    prediction_records: dict[str, list[dict]] = {}
    model_provenance = {}
    torch.set_num_threads(4)
    for condition, directory in CHECKPOINTS.items():
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        checkpoint = directory / "best.pt"
        normalization_path = directory / "normalization.npz"
        checkpoint_sha = sha256(checkpoint)
        if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256[condition]:
            raise ValueError(f"{condition} checkpoint SHA-256 differs from frozen reference")
        normalization = np.load(normalization_path, allow_pickle=False)
        normalization_sha = sha256(normalization_path)
        if normalization_sha != EXPECTED_NORMALIZATION_SHA256[condition]:
            raise ValueError(f"{condition} normalization SHA-256 differs from frozen reference")
        validation_ids_path = directory / "validation_ids.json"
        if sha256(validation_ids_path) != EXPECTED_INTERNAL_VALIDATION_IDS_SHA256:
            raise ValueError(f"{condition} internal validation IDs differ from frozen reference")
        if condition == "question_only":
            q = (questions - normalization["mean"]) / normalization["std"]
            candidate_features = build_features(np.zeros_like(q), q, condition)
        else:
            a = (question_audio - normalization["audio_mean"]) / normalization["audio_std"]
            q = (questions - normalization["question_mean"]) / normalization["question_std"]
            candidate_features = build_features(a, q, condition)
        model = FrozenFeatureCandidateScorer(770, 64)
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            raw_logits = model(candidate_features).float()
            scaled_logits = raw_logits / float(result["temperature"])
            probabilities = scaled_logits.softmax(dim=-1).numpy()
        if probabilities.shape != (244, 2) or not np.isfinite(probabilities).all():
            raise ValueError(f"invalid probabilities from {condition}")
        conditions[condition] = probabilities[:, 1]
        prediction_records[condition] = [
            {
                "row": index,
                "source_id": row["source_id"],
                "normalized_question": row["normalized_question"],
                "target": int(targets[index]),
                "prediction": int(probabilities[index].argmax()),
                "raw_logits": [float(x) for x in raw_logits[index]],
                "temperature_scaled_logits": [float(x) for x in scaled_logits[index]],
                "probabilities": [float(x) for x in probabilities[index]],
                "votes": row["votes"],
                "confidence_values": row["confidence_values"],
                "uploader": row["uploader"],
            }
            for index, row in enumerate(manifest)
        ]
        model_provenance[condition] = {
            "checkpoint_sha256": checkpoint_sha,
            "normalization_sha256": normalization_sha,
            "temperature": float(result["temperature"]),
            "development_validation_metrics": result["metrics"],
        }

    groups = [row["uploader"] for row in manifest]
    scores = {}
    for name, values in conditions.items():
        scores[name] = binary_metrics(targets, values)
        scores[name]["mean_confidence"] = float(np.maximum(values, 1 - values).mean())
        scores[name]["disagreement_count"] = int(sum(row["disagreement"] for row in manifest))
        scores[name]["disagreement_rate"] = float(
            np.mean([row["disagreement"] for row in manifest])
        )
    bootstrap = paired_cluster_bootstrap(targets, conditions, groups)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    for name, records in prediction_records.items():
        (OUTPUT / f"{name}-predictions.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
    report = {
        "status": "PASS",
        "scope": (
            "one-shot frozen external validation; no training, fitting, selection, or calibration"
        ),
        "cohort": {
            "rows": len(manifest),
            "unique_audio_assets": len(audio_ids),
            "uploader_groups": len(set(groups)),
            "manifest_sha256": sha256(manifest_path),
            "publisher_validation_csv_sha256": sha256(ATTEMPT / "clotho_aqa_val.csv"),
            "publisher_validation_csv_md5": "41b8f6d906a57d7a5a4ee02bbcdefcc9",
            "total_positive": int(targets.sum()),
            "total_negative": int(len(targets) - targets.sum()),
            "disagreement_count": int(sum(row["disagreement"] for row in manifest)),
            "feature_report_sha256": sha256(features_dir / "result.json"),
        },
        "conditions": model_provenance,
        "metrics": scores,
        "paired_uploader_bootstrap": bootstrap,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": "cpu",
            "command": "python -m scripts.score_clotho_publisher_val",
            "seconds": time.perf_counter() - started,
            "gpu_used": False,
        },
        "limitations": [
            "The preregistered 200 independent-uploader floor was not met; "
            "metrics are descriptive only.",
            "This is an observed validation set, not a blind or sealed audit set.",
            "The two fixed checkpoints were selected using the existing development corpus; "
            "no selection was performed here.",
            "EfficientAT pretraining item overlap could not be established from released "
            "training IDs.",
        ],
    }
    (OUTPUT / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
