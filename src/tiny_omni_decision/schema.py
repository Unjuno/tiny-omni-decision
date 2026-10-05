from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class TextDecision(BaseModel):
    state: str
    question: str
    options: list[str] = Field(min_length=2, max_length=62)
    target: str

    @model_validator(mode="after")
    def target_is_an_option(self) -> TextDecision:
        if self.target not in self.options:
            raise ValueError("target must match one of the options")
        if len(set(self.options)) != len(self.options):
            raise ValueError("options must be unique")
        return self


class MediaRef(BaseModel):
    kind: Literal["image", "audio", "video"]
    path: str | None = None
    uri: str | None = None
    frame_refs: list[str] = Field(default_factory=list)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    license: str | None = None

    @model_validator(mode="after")
    def has_reference(self) -> MediaRef:
        if bool(self.path) == bool(self.uri):
            raise ValueError("media needs exactly one relative path or remote URI")
        if self.path and (
            PurePosixPath(self.path).is_absolute()
            or PureWindowsPath(self.path).is_absolute()
            or ".." in self.path.replace("\\", "/").split("/")
        ):
            raise ValueError("media path must be relative and remain inside the data root")
        return self


class LicenseProvenance(BaseModel):
    license: str = "UNKNOWN"
    commercial_use: bool | None = None
    derivative_model_training_allowed: bool | None = None
    redistribution_allowed: bool | None = None
    media_redistribution_allowed: bool | None = None
    attribution: str | None = None
    source_component: str | None = None
    trust_status: Literal["trusted", "review", "untrusted"] = "review"


class DecisionExample(BaseModel):
    id: str = Field(min_length=1)
    modality: Literal["text", "image", "audio", "video"]
    state: str
    question: str
    options: list[str] = Field(min_length=2, max_length=62)
    target: str
    media: list[MediaRef] = Field(default_factory=list)
    source: str = Field(min_length=1)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_record_id: str = Field(min_length=1)
    split: str
    task_type: Literal[
        "temporal_descriptive",
        "static_descriptive",
        "explanatory",
        "predictive",
        "counterfactual",
    ] | None = None
    task_group_id: str | None = None
    source_target: Any | None = None
    provenance: LicenseProvenance

    @model_validator(mode="after")
    def validate_decision(self) -> DecisionExample:
        if self.target not in self.options:
            raise ValueError("target must match one of the options")
        if len(set(self.options)) != len(self.options):
            raise ValueError("options must be unique")
        if self.modality != "text" and not any(item.kind == self.modality for item in self.media):
            raise ValueError("non-text examples must include media matching their modality")
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


class DatasetSourceComponent(BaseModel):
    component_id: str
    url: str | None = None
    license: str = "UNKNOWN"
    commercial_use: bool | None = None
    derivative_model_training_allowed: bool | None = None
    redistribution_allowed: bool | None = None
    media_redistribution_allowed: bool | None = None
    attribution: str | None = None
    modalities: list[str] = Field(default_factory=list)
    trust_status: Literal["trusted", "review", "untrusted"] = "review"


class DatasetManifest(BaseModel):
    schema_version: int = 1
    dataset_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    subset: str | None = None
    split: str
    modalities: list[str] = Field(min_length=1)
    upstream_url: str
    license: str
    commercial_use: bool | None = None
    derivative_model_training_allowed: bool | None = None
    redistribution_allowed: bool | None = None
    media_redistribution_allowed: bool | None = None
    attribution: str | None = None
    source_citation: str | None = None
    usage: Literal["training", "evaluation", "both"]
    trust_status: Literal["trusted", "review", "untrusted"] = "review"
    source_components: list[DatasetSourceComponent] = Field(default_factory=list)
    schema_map: DatasetSchemaMap | None = Field(default=None, alias="schema")
    notes: dict[str, Any] = Field(default_factory=dict)


class DatasetCatalogEntry(BaseModel):
    manifest: str
    include: bool
    subset: str | None = None
    split: str | None = None
    adapter: str | None = None
    row_limit: int | None = Field(default=None, ge=1)
    heldout_partition: Literal["validation", "evaluation", "split"] | None = None
    reason: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def validate_entry(self) -> DatasetCatalogEntry:
        path = PurePosixPath(self.manifest.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("catalog manifest path must stay inside the repository")
        if self.include and not self.adapter:
            raise ValueError("included catalog entries need an adapter")
        if not self.include and not (self.reason or self.note):
            raise ValueError("excluded catalog entries need a reason")
        return self


class DatasetCatalog(BaseModel):
    schema_version: int = 1
    manifest_type: Literal["candidate_catalog"]
    purpose: Literal["training", "evaluation"]
    sources: list[DatasetCatalogEntry] = Field(min_length=1)
