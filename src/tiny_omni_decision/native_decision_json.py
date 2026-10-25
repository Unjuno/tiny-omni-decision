"""Serialize typed native-observation decisions without generating option text."""

import math
from numbers import Real

from tiny_omni_decision.observations import ObservationBatch


def native_decision_json(batch: ObservationBatch, probabilities, *, abstain_reason=None):
    """Return a stable JSON-ready decision over the request's typed options.

    Probabilities are positional and must match ``batch.options`` exactly.
    Abstentions carry null probabilities instead of an invented distribution.
    Exact probability ties select the lexicographically smallest option ID.
    """
    if isinstance(batch, ObservationBatch):
        batch = ObservationBatch.model_validate(batch.model_dump(mode="json"))
    else:
        batch = ObservationBatch.model_validate(batch)

    options = [
        {"option_id": option.option_id, "attributes": option.attributes}
        for option in batch.options
    ]
    result = {
        "schema_version": "1.0",
        "present_modalities": list(batch.present_modalities),
    }
    if abstain_reason is not None:
        if (
            probabilities is not None
            or not isinstance(abstain_reason, str)
            or not abstain_reason.strip()
        ):
            raise ValueError("abstention requires a reason and no probability vector")
        result.update(
            status="abstain",
            selected_option_id=None,
            confidence=None,
            reason=abstain_reason,
            options=[dict(option, probability=None) for option in options],
        )
        return result

    if probabilities is None or isinstance(probabilities, (str, bytes)):
        raise ValueError("probabilities must match the supplied options")
    try:
        values = list(probabilities)
    except TypeError as error:
        raise ValueError("probabilities must match the supplied options") from error
    if len(values) != len(options) or any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
        or value < 0 or value > 1
        for value in values
    ) or not math.isclose(math.fsum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("finite normalized probabilities aligned to all options required")

    values = [float(value) for value in values]
    selected_index = min(
        range(len(options)), key=lambda index: (-values[index], options[index]["option_id"])
    )
    result.update(
        status="selected",
        selected_option_id=options[selected_index]["option_id"],
        confidence=values[selected_index],
        options=[
            dict(option, probability=value)
            for option, value in zip(options, values, strict=True)
        ],
    )
    return result
