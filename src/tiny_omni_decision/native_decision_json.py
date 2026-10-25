"""Serialize typed native-observation decisions without generating option text."""

import math
from numbers import Real

from tiny_omni_decision.observations import ObservationBatch


def native_classification_request(batch, modality):
    """Validate a single-modality text-free classification request."""
    if isinstance(batch, ObservationBatch):
        batch = ObservationBatch.model_validate(batch.model_dump(mode="json"))
    else:
        batch = ObservationBatch.model_validate(batch)
    if len(batch.observations) != 1 or batch.observations[0].modality != modality:
        raise ValueError(f"one {modality} observation is required")
    if batch.criterion is not None:
        raise ValueError("fixed-taxonomy native classification does not accept a criterion")
    class_ids = []
    for option in batch.options:
        if set(option.attributes) != {"class_id"}:
            raise ValueError("native classification options require only class_id")
        class_id = option.attributes["class_id"]
        if not isinstance(class_id, str) or not class_id.strip():
            raise ValueError("native classification class IDs must be nonempty strings")
        class_ids.append(class_id)
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("native classification class IDs must be unique")
    reason = None
    if not all(batch.observations[0].valid_mask):
        reason = f"incomplete_{modality}_observation"
    return batch, class_ids, reason


def native_decision_json(
    batch: ObservationBatch,
    probabilities,
    *,
    selected_option_id=None,
    abstain_reason=None,
):
    """Return a stable JSON-ready decision over the request's typed options.

    Probabilities are positional and must match ``batch.options`` exactly.
    Abstentions carry null probabilities instead of an invented distribution.
    Exact probability ties default to the lexicographically smallest option ID;
    a model's own tied selection may be supplied explicitly.
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
        if (probabilities is not None or selected_option_id is not None
                or not isinstance(abstain_reason, str) or not abstain_reason.strip()):
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
    maximum_probability = max(values)
    selected_index = min(
        range(len(options)), key=lambda index: (-values[index], options[index]["option_id"])
    )
    if selected_option_id is not None:
        option_ids = [option["option_id"] for option in options]
        if selected_option_id not in option_ids:
            raise ValueError("selected option ID is not supplied")
        selected_index = option_ids.index(selected_option_id)
        if values[selected_index] != maximum_probability:
            raise ValueError("selected option must have the maximum probability")
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
