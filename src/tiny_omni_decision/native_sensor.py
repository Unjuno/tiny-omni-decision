"""Small native sensor task probe; no text model, fixed HARTH activity semantics."""

import torch
from torch import nn

from tiny_omni_decision.observations import ObservationBatch
from tiny_omni_decision.sensor_payload import load_sensor_payload

CHANNELS = ("back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z")
ACTIVITY_CODES = (1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140)


class NativeSensorDecision(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(7, 32, 5, padding=2), nn.GELU(), nn.Conv1d(32, 32, 5, padding=2), nn.GELU()
        )
        self.head = nn.Linear(32, len(ACTIVITY_CODES))

    def forward(self, values, timestamps_ms):
        if values.ndim != 3 or values.shape[-1] != 6 or timestamps_ms.shape != values.shape[:2]:
            raise ValueError("expected aligned [batch,time,6] sensor input")
        if (
            values.shape[1] < 2
            or not torch.isfinite(values).all()
            or not torch.isfinite(timestamps_ms).all()
        ):
            raise ValueError("finite multi-sample input required")
        deltas = timestamps_ms[:, 1:] - timestamps_ms[:, :-1]
        if (deltas <= 0).any() or (deltas > 30).any():
            raise ValueError("noncontiguous timestamps")
        dt = torch.cat((torch.zeros_like(deltas[:, :1]), deltas / 20.0), dim=1)
        features = torch.cat((values, dt.unsqueeze(-1).to(values.dtype)), dim=-1)
        return self.head(self.stem(features.transpose(1, 2)).mean(-1))


def decide_sensor(model: NativeSensorDecision, batch: ObservationBatch):
    batch = ObservationBatch.model_validate(batch.model_dump())
    if len(batch.observations) != 1 or batch.observations[0].modality != "sensor":
        raise ValueError("this bounded probe accepts one sensor observation only")
    if batch.criterion is not None and not (
        batch.criterion.kind == "typed"
        and batch.criterion.value == "harth_activity"
        and not batch.criterion.parameters
    ):
        raise ValueError("unsupported criterion for activity probe")
    observation = batch.observations[0]
    if tuple(c.name for c in observation.channels) != CHANNELS:
        raise ValueError("unexpected channel ordering")
    if any(c.unit != "g" for c in observation.channels):
        raise ValueError("expected calibrated acceleration in g")
    codes = []
    for option in batch.options:
        if set(option.attributes) != {"activity_code"}:
            raise ValueError("unsupported option attributes")
        code = option.attributes["activity_code"]
        if type(code) is not int or code not in ACTIVITY_CODES or code in codes:
            raise ValueError("activity codes must be supported unique integers")
        codes.append(code)
    payload = load_sensor_payload(observation)
    if not payload.valid_mask.all():
        return dict(
            schema_version="1.0",
            status="abstain",
            selected_option_id=None,
            reason="incomplete_sensor_window",
            options=[dict(option_id=o.option_id, probability=None) for o in batch.options],
        )
    device = next(model.parameters()).device
    values = torch.tensor(payload.values, device=device, dtype=torch.float32)[None]
    times = torch.tensor(payload.timestamps_ms, device=device, dtype=torch.float64)[None]
    if model.training:
        raise ValueError("inference requires eval mode")
    with torch.inference_mode():
        logits = model(values, times)[0].double()
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite logits")
        ordered = sorted(codes)
        p = logits[[ACTIVITY_CODES.index(c) for c in ordered]].softmax(0).cpu().tolist()
    probabilities = dict(zip(ordered, p, strict=True))
    # Stable semantic-code tie break, independent of supplied option order or IDs.
    selected = max(ordered, key=probabilities.__getitem__)
    return dict(
        schema_version="1.0",
        status="selected",
        selected_option_id=next(
            o.option_id for o, c in zip(batch.options, codes, strict=True) if c == selected
        ),
        options=[
            dict(option_id=o.option_id, probability=probabilities[c])
            for o, c in zip(batch.options, codes, strict=True)
        ],
    )
