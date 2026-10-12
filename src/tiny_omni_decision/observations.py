"""Versioned, text-optional observation metadata; no encoder imports or inference."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictBool, model_validator

Nonempty = Annotated[str, Field(min_length=1, pattern=r".*\S.*")]
Modality = Literal["text", "audio", "image", "screen", "video", "sensor"]
Scalar = StrictBool | int | FiniteFloat | str


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SensorChannel(Contract):
    name: Nonempty
    unit: Nonempty
    scale: FiniteFloat
    offset: FiniteFloat
    valid_range: tuple[FiniteFloat, FiniteFloat] | None = None

    @model_validator(mode="after")
    def check_calibration(self):
        if self.scale == 0:
            raise ValueError("calibration scale must be nonzero")
        if self.valid_range is not None and self.valid_range[0] >= self.valid_range[1]:
            raise ValueError("channel range must increase")
        return self


class Observation(Contract):
    modality: Modality
    source_id: Nonempty
    payload_ref: Nonempty
    payload_index: Annotated[int, Field(ge=0, strict=True)] | None = None
    payload_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    clock_id: Nonempty
    timestamps_ms: tuple[FiniteFloat, ...]
    valid_mask: tuple[StrictBool, ...]
    channels: tuple[SensorChannel, ...] = ()

    @model_validator(mode="after")
    def check_alignment(self):
        if not self.timestamps_ms or len(self.timestamps_ms) != len(self.valid_mask):
            raise ValueError("timestamps and validity must be nonempty and aligned")
        if any(b <= a for a, b in zip(self.timestamps_ms, self.timestamps_ms[1:], strict=False)):
            raise ValueError("timestamps must strictly increase")
        if self.modality == "sensor" and not self.channels:
            raise ValueError("sensor channels require units and calibration")
        if self.modality != "sensor" and self.channels:
            raise ValueError("sensor channels on non-sensor observation")
        if len({c.name for c in self.channels}) != len(self.channels):
            raise ValueError("duplicate sensor channels")
        return self


class NativeOption(Contract):
    option_id: Nonempty
    attributes: dict[str, Scalar]


class Criterion(Contract):
    kind: Literal["typed", "text", "spoken"]
    value: Nonempty
    parameters: dict[str, Scalar] = Field(default_factory=dict)
    # typed: value is application criterion code; text: literal condition;
    # spoken: value is the source_id of an audio observation in this batch.


class ObservationBatch(Contract):
    schema_version: Literal["1.0"]
    observations: tuple[Observation, ...]
    options: tuple[NativeOption, ...]
    criterion: Criterion | None = None

    @property
    def present_modalities(self):
        return tuple(sorted({o.modality for o in self.observations}))

    @model_validator(mode="after")
    def check_batch(self):
        if not self.observations:
            raise ValueError("at least one observation is required")
        if len({o.source_id for o in self.observations}) != len(self.observations):
            raise ValueError("source IDs must be unique within request")
        if len(self.options) < 2 or len({o.option_id for o in self.options}) != len(self.options):
            raise ValueError("at least two unique option IDs are required")
        if self.criterion and self.criterion.kind == "spoken":
            if not any(
                o.source_id == self.criterion.value and o.modality == "audio"
                for o in self.observations
            ):
                raise ValueError("spoken criterion must reference present audio")
        return self

    def require_shared_clock(self):
        """Fusion may call this; asynchronous clocks require explicit upstream alignment."""
        if len({o.clock_id for o in self.observations}) != 1:
            raise ValueError("cross-clock fusion requires an explicit alignment transform")
