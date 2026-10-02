from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class TextDecision(BaseModel):
    state: str
    question: str
    options: list[str] = Field(min_length=2, max_length=26)
    target: str

    @model_validator(mode="after")
    def target_is_an_option(self) -> TextDecision:
        if self.target not in self.options:
            raise ValueError("target must match one of the options")
        if len(set(self.options)) != len(self.options):
            raise ValueError("options must be unique")
        return self


class BaseModelManifest(BaseModel):
    schema_version: int = 1
    repo_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    role: str
    license: str
    modalities: list[str] = Field(min_length=1)
    processor_repo_id: str | None = None
    processor_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    files: list[dict[str, Any]] = Field(default_factory=list)
    notes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def revisions_match(self) -> BaseModelManifest:
        if self.processor_revision != self.revision:
            raise ValueError("processor revision must match the pinned base-model revision")
        return self


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
    schema_map: DatasetSchemaMap = Field(alias="schema")
    notes: dict[str, Any] = Field(default_factory=dict)
