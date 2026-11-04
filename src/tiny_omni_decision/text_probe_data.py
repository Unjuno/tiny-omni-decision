from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .dataset import iter_local_rows, normalize_jsonl
from .schema import DatasetManifest, DecisionExample


def load_generic_text_examples(
    path: Path, manifest: DatasetManifest
) -> list[DecisionExample]:
    """Load normalized generic text examples without importing ML runtimes."""
    return list(normalize_jsonl(iter_local_rows(path), manifest, "generic"))


def validate_generic_probe_manifests(
    training: DatasetManifest, evaluation: DatasetManifest
) -> None:
    if training.split != "train" or training.usage != "training":
        raise ValueError("generic probe --manifest must be a training-split manifest")
    if (
        evaluation.dataset_id != training.dataset_id
        or evaluation.revision != training.revision
        or evaluation.split != "validation"
        or evaluation.usage != "evaluation"
    ):
        raise ValueError("generic probe requires a matching, independent evaluation manifest")


def text_probe_identity_sets(
    examples: list[DecisionExample],
) -> tuple[set[str], set[str]]:
    """Return source-group and normalized-content identities for leakage checks."""
    groups = {example.source_record_id.split(":", 1)[0] for example in examples}
    content = {
        hashlib.sha256(
            json.dumps(
                {
                    "state": example.state,
                    "question": example.question,
                    "options": sorted(example.options),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            .encode("utf-8")
        ).hexdigest()
        for example in examples
    }
    return groups, content
