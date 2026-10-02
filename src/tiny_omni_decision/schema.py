from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class BaseModelManifest(BaseModel):
    schema_version: int = 1
    repo_id: str
    revision: str | None = None
    role: str
    license: str
    modalities: list[str] = Field(min_length=1)
    processor_repo_id: str | None = None
    processor_revision: str | None = None
    files: list[dict[str, Any]] = Field(default_factory=list)
    notes: dict[str, Any] = Field(default_factory=dict)


class DatasetSchemaMap(BaseModel):
    state: str
    question: str
    options: str
    target: str
    media: str | None = None


class DatasetManifest(BaseModel):
    schema_version: int = 1
    dataset_id: str
    revision: str | None = None
    subset: str | None = None
    split: str
    modalities: list[str] = Field(min_length=1)
    license: str
    commercial_use: bool | None = None
    redistribution_allowed: bool | None = None
    media_redistribution_allowed: bool | None = None
    usage: Literal["training", "evaluation", "both"]
    schema: DatasetSchemaMap
    notes: dict[str, Any] = Field(default_factory=dict)
