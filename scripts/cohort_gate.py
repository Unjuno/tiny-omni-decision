"""Fail-closed, source-scoped cohort gate for new R2 model runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.check_cohort_usage import check_cohort, group_digest

DEFAULT_LEDGER = (
    Path(__file__).resolve().parents[1]
    / "manifests"
    / "pretrained_reuse"
    / "cohort_usage_ledger.json"
)


class CohortUseBlocked(RuntimeError):
    """Raised before media/model loading when cohort history is not clear."""


def require_clear_cohort(
    *, ledger_path: Path, source_id: str, revision: str, group_ids: list[str]
) -> dict[str, Any]:
    """Allow a run only when its source history is complete and groups are new."""
    ledger_bytes = ledger_path.read_bytes()
    ledger = json.loads(ledger_bytes)
    result = check_cohort(
        ledger,
        source_id=source_id,
        revision=revision,
        group_ids=group_ids,
    )
    if result["decision"] != "CLEAR_WITHIN_DECLARED_HISTORY":
        raise CohortUseBlocked(
            "cohort use gate blocked before model/media loading: "
            f"{result['decision']} for {source_id}@{revision}"
        )
    result["ledger_sha256"] = hashlib.sha256(ledger_bytes).hexdigest()
    result["proposed_group_identity_sha256"] = sorted(
        group_digest(source_id, revision, group_id) for group_id in group_ids
    )
    return result
