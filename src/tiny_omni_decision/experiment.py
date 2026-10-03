from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ExperimentManifest(BaseModel):
    """Serializable provenance and results for one attempted Teacher run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    experiment_id: str = Field(min_length=1)
    status: Literal["started", "completed", "failed"]
    seed: int = Field(ge=0)
    base_model_repo_id: str | None = None
    base_model_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    base_model_weights_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    train_corpus_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    validation_corpus_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    sealed_audit_corpus_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    sampling_policy: dict[str, Any] = Field(default_factory=dict)
    optimizer_schedule: dict[str, Any] = Field(default_factory=dict)
    learning_curve: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifact: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    failure: str | None = None


def append_experiment_event(path: Path, event: dict[str, Any]) -> None:
    """Append one immutable lifecycle event for every attempted run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
