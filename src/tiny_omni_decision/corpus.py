from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from .schema import DecisionExample, MediaRef


def media_identity(media: MediaRef) -> str:
    """Return a stable identity for one underlying media asset."""
    kind = media.kind
    digest = media.sha256
    if digest:
        return f"{kind}:sha256:{digest}"
    uri = media.uri
    path = media.path
    reference = unquote(str(uri or path or "")).replace("\\", "/").casefold()
    return f"{kind}:{reference}"


def _state_identity(example: DecisionExample) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", example.state).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def source_asset_identity(example: DecisionExample) -> str:
    """Identify a source-level unit that must remain within a single split."""
    if example.source_asset_group_id:
        return f"source-asset-group:{example.source_asset_group_id}"
    if example.source == "physionpp/readout" and example.task_group_id:
        return f"scene-seed:{example.task_group_id}"
    if example.source == "n4ze3m/typed-decisions-synth":
        return f"state:{example.source_record_id.split(':', 1)[0]}"
    if example.modality in {"image", "video"} and example.media:
        return media_identity(example.media[0])
    if example.modality == "audio" and example.media:
        references = [
            example.source_record_id,
            example.media[0].uri or "",
            example.media[0].path or "",
        ]
        for reference in references:
            decoded = unquote(str(reference)).replace("\\", "/")
            basename = PurePosixPath(decoded).name
            match = re.search(r"([^/]+?)_nohash_[^/]+(?:\.wav)?$", basename, re.IGNORECASE)
            if match:
                return f"speaker:{match.group(1).casefold()}"
        return media_identity(example.media[0])
    if example.modality == "text" and example.state.strip():
        return f"state:{_state_identity(example)}"
    return f"record:{example.source_record_id}"


def filter_previously_seen_records(
    candidates: list[DecisionExample], previously_seen: list[DecisionExample]
) -> tuple[list[DecisionExample], dict[str, object]]:
    """Drop candidate source-asset groups found in a prior corpus.

    This is used before freezing a new validation or audit partition. A single
    previously observed question removes its whole image, speaker, video, or
    text-state group from the candidate partition.
    """
    from collections import defaultdict

    from .dataset import content_fingerprint

    used_ids = {(item.source, item.source_record_id) for item in previously_seen}
    used_assets = {(item.source, source_asset_identity(item)) for item in previously_seen}
    used_media = {media_identity(reference) for item in previously_seen for reference in item.media}
    used_content = {content_fingerprint(item) for item in previously_seen}

    groups: dict[tuple[str, str], list[DecisionExample]] = defaultdict(list)
    for item in candidates:
        groups[(item.source, source_asset_identity(item))].append(item)

    excluded_groups: dict[tuple[str, str], str] = {}
    for key, records in groups.items():
        if any((item.source, item.source_record_id) in used_ids for item in records):
            excluded_groups[key] = "source_record_id"
        elif key in used_assets:
            excluded_groups[key] = "source_asset"
        elif any(
            media_identity(reference) in used_media for item in records for reference in item.media
        ):
            excluded_groups[key] = "media_identity"
        elif any(content_fingerprint(item) in used_content for item in records):
            excluded_groups[key] = "normalized_content"

    retained = [
        item
        for item in candidates
        if (item.source, source_asset_identity(item)) not in excluded_groups
    ]
    excluded_by_reason: dict[str, int] = defaultdict(int)
    for key, reason in excluded_groups.items():
        excluded_by_reason[reason] += len(groups[key])
    return retained, {
        "candidate_records": len(candidates),
        "retained_records": len(retained),
        "excluded_records": len(candidates) - len(retained),
        "candidate_asset_groups": len(groups),
        "excluded_asset_groups": len(excluded_groups),
        "excluded_by_reason": dict(sorted(excluded_by_reason.items())),
    }


def validate_training_inputs(
    train_path: str | Path,
    validation_path: str | Path | None,
    *,
    evaluation_path: str | Path | None = None,
) -> None:
    """Fail closed unless training and selection validation are independent inputs."""
    if validation_path is None:
        raise ValueError("independent validation is required for checkpoint selection")
    if evaluation_path is not None:
        raise ValueError("evaluation is isolated from normal experiment iteration")
    train = Path(train_path).resolve()
    validation = Path(validation_path).resolve()
    if train == validation:
        raise ValueError("training and validation must not use the same corpus")
    for path in (train, validation):
        components = {part.casefold().replace("_", "-") for part in path.parts}
        if any("sealed-audit" in part for part in components):
            raise ValueError("sealed audit data cannot be loaded during training")


def macro_metrics(metrics: dict[str, dict[str, float | int]]) -> dict[str, float]:
    """Aggregate each modality equally, independently of its sample count."""
    modality_metrics = [metrics[key] for key in sorted(metrics) if key.startswith("modality:")]
    if not modality_metrics:
        raise ValueError("macro metrics require at least one modality")
    accuracy = [float(item["accuracy"]) for item in modality_metrics]
    return {
        "macro_accuracy": sum(accuracy) / len(accuracy),
        "minimum_modality_accuracy": min(accuracy),
        "macro_nll": sum(float(item["nll"]) for item in modality_metrics) / len(modality_metrics),
        "macro_brier": sum(float(item["brier"]) for item in modality_metrics)
        / len(modality_metrics),
        "macro_ece": sum(float(item["ece"]) for item in modality_metrics) / len(modality_metrics),
    }


def validation_selection_score(metrics: dict[str, dict[str, float | int]]) -> float:
    """Predefined validation-only score; balance calibration and weak-modality quality."""
    summary = macro_metrics(metrics)
    return (
        summary["macro_nll"]
        + 0.2 * summary["macro_brier"]
        + 0.1 * summary["macro_ece"]
        - 0.25 * summary["macro_accuracy"]
        - 0.25 * summary["minimum_modality_accuracy"]
    )


@dataclass
class ValidationCheckpointSelector:
    """Track best validation checkpoint and stop after a fixed plateau patience."""

    patience: int = 4
    min_delta: float = 0.0
    best_score: float = float("inf")
    best_step: int = 0
    evaluations_without_improvement: int = 0

    def __post_init__(self) -> None:
        if self.patience < 1 or self.min_delta < 0:
            raise ValueError("patience must be positive and min_delta nonnegative")

    def observe(self, step: int, metrics: dict[str, dict[str, float | int]]) -> bool:
        if step < 1:
            raise ValueError("checkpoint step must be positive")
        score = validation_selection_score(metrics)
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.best_step = step
            self.evaluations_without_improvement = 0
            return True
        self.evaluations_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.best_step > 0 and self.evaluations_without_improvement >= self.patience


def _record_group(example: DecisionExample) -> tuple[str, str, str]:
    return example.source, example.split, source_asset_identity(example)


def partition_heldout_records(
    examples: list[DecisionExample],
    *,
    seed: int,
    validation_fraction: float = 0.5,
    heldout_partitions: dict[tuple[str, str], str] | None = None,
) -> tuple[list[DecisionExample], list[DecisionExample]]:
    """Keep declared upstream splits fixed and partition remaining records deterministically."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    groups: dict[tuple[str, str, str], list[DecisionExample]] = defaultdict(list)
    for example in examples:
        groups[_record_group(example)].append(example)
    roles = heldout_partitions or {}
    source_splits = {(source, split) for source, split, _ in groups}
    if heldout_partitions is not None:
        missing = source_splits - roles.keys()
        if missing:
            raise ValueError(
                f"held-out source splits need explicit partition roles: {sorted(missing)}"
            )
    invalid = {role for role in roles.values()} - {"validation", "evaluation", "split"}
    if invalid:
        raise ValueError(f"unsupported held-out partition roles: {sorted(invalid)}")

    groups_by_source_split: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for group in groups:
        groups_by_source_split[group[:2]].append(group)

    validation_groups: set[tuple[str, str, str]] = set()
    evaluation_groups: set[tuple[str, str, str]] = set()
    for source_split, group_ids in sorted(groups_by_source_split.items()):
        source, split = source_split
        role = roles.get(source_split, "split")
        if role == "validation":
            validation_groups.update(group_ids)
            continue
        if role == "evaluation":
            evaluation_groups.update(group_ids)
            continue
        ordered = sorted(group_ids)
        random.Random(f"{seed}:{source}:{split}:heldout-partition").shuffle(ordered)
        if len(ordered) < 2:
            raise ValueError(
                f"source {source} split {split} needs at least two held-out records to split"
            )
        validation_count = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
        validation_groups.update(ordered[:validation_count])
        evaluation_groups.update(ordered[validation_count:])

    validation = [example for example in examples if _record_group(example) in validation_groups]
    evaluation = [example for example in examples if _record_group(example) in evaluation_groups]
    return validation, evaluation


def comparison_deltas(
    baseline: dict[str, dict[str, float | int]],
    trained: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float]]:
    """Return trained-minus-base metric changes for shared metrics in each group."""
    delta: dict[str, dict[str, float]] = {}
    for group in sorted(baseline.keys() & trained.keys()):
        delta[group] = {
            metric: float(trained[group][metric]) - float(baseline[group][metric])
            for metric in sorted((baseline[group].keys() & trained[group].keys()) - {"count"})
        }
    return delta


def selection_loss(metrics: dict[str, float | int], *, brier_weight: float = 0.2) -> float:
    return float(metrics["nll"]) + brier_weight * float(metrics["brier"])


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
