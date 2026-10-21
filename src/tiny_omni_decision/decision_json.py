"""Dependency-light typed JSON contract for supplied-option decisions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real


def decision_json_record(
    *,
    request_id: str,
    decision_type: str,
    option_ids: Sequence[str],
    option_texts: Sequence[str],
    probabilities: Sequence[Real],
    predicted_index: int | None = None,
    confidence: Real | None = None,
    ordinal_values: Sequence[Real] | None = None,
) -> dict[str, object]:
    """Build a validated, JSON-ready Decision record without importing a model stack."""

    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a nonempty string")
    if not isinstance(decision_type, str) or decision_type not in {
        "choice", "yes_no", "ordinal_score"
    }:
        raise ValueError("unsupported decision_type")
    if any(isinstance(values, (str, bytes)) for values in (option_ids, option_texts)):
        raise ValueError("option IDs and texts must be sequences of strings")
    if len(option_ids) < 2 or len(option_ids) != len(option_texts):
        raise ValueError("option IDs and texts must have matching lengths of at least two")
    if (
        any(not isinstance(value, str) or not value.strip() for value in option_ids)
        or len(set(option_ids)) != len(option_ids)
    ):
        raise ValueError("option IDs must be nonempty unique strings")
    if any(not isinstance(value, str) or not value.strip() for value in option_texts):
        raise ValueError("option texts must be nonempty strings")
    if decision_type == "yes_no" and len(option_ids) != 2:
        raise ValueError("yes_no decisions require exactly two options")
    if decision_type == "ordinal_score":
        if ordinal_values is None or len(ordinal_values) != len(option_ids):
            raise ValueError("ordinal_score requires one numeric value per option")
        numeric_ordinal_values = _finite_real_values(ordinal_values, "ordinal values")
        if any(left >= right for left, right in zip(
            numeric_ordinal_values[:-1], numeric_ordinal_values[1:], strict=True
        )):
            raise ValueError("ordinal values must be finite and strictly increasing")
    elif ordinal_values is not None:
        raise ValueError("ordinal_values are only valid for ordinal_score decisions")
    if isinstance(probabilities, (str, bytes)) or len(probabilities) != len(option_ids):
        raise ValueError("probability vector must match the supplied option count")
    values = _finite_real_values(probabilities, "probabilities")
    if any(value < 0 or value > 1 for value in values) or not math.isclose(
        sum(values), 1.0, abs_tol=1e-5
    ):
        raise ValueError("invalid probability distribution")

    selected_index = max(range(len(values)), key=values.__getitem__)
    if predicted_index is None:
        predicted_index = selected_index
    elif (
        isinstance(predicted_index, bool)
        or not isinstance(predicted_index, int)
        or not 0 <= predicted_index < len(values)
    ):
        raise ValueError("predicted_index is outside the supplied option range")
    if not math.isclose(
        values[predicted_index], values[selected_index], rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("predicted_index does not match the probability maximum")

    selected_probability = values[predicted_index]
    if confidence is None:
        confidence_value = selected_probability
    else:
        confidence_values = _finite_real_values([confidence], "confidence")
        confidence_value = confidence_values[0]
        if not math.isclose(confidence_value, selected_probability, abs_tol=1e-5):
            raise ValueError("confidence does not match the selected-option probability")

    record: dict[str, object] = {
        "schema_version": 1,
        "request_id": request_id,
        "decision_type": decision_type,
        "selected_option_id": option_ids[predicted_index],
        "confidence": confidence_value,
        "options": [
            {"id": option_id, "text": option_text, "probability": probability}
            for option_id, option_text, probability in zip(
                option_ids, option_texts, values, strict=True
            )
        ],
    }
    if ordinal_values is not None:
        record["expected_score"] = sum(
            probability * score
            for probability, score in zip(values, numeric_ordinal_values, strict=True)
        )
    return record


def decision_json_abstention_record(
    *,
    request_id: str,
    decision_type: str,
    option_ids: Sequence[str],
    option_texts: Sequence[str],
    reason: str,
) -> dict[str, object]:
    """Build a JSON record for a decision that intentionally has no distribution."""

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("abstention reason must be a nonempty string")
    if not isinstance(decision_type, str) or decision_type not in {"choice", "yes_no"}:
        raise ValueError("abstention supports choice and yes_no decisions only")
    # Reuse the selected-record validator for request and option identity checks,
    # then replace its synthetic uniform probabilities with explicit nulls.
    record = decision_json_record(
        request_id=request_id,
        decision_type=decision_type,
        option_ids=option_ids,
        option_texts=option_texts,
        probabilities=(
            [1.0 / len(option_ids)] * len(option_ids) if len(option_ids) else []
        ),
    )
    record.update(
        status="abstain",
        selected_option_id=None,
        confidence=None,
        reason=reason,
        options=[
            {"id": option_id, "text": option_text, "probability": None}
            for option_id, option_text in zip(option_ids, option_texts, strict=True)
        ],
    )
    return record


def _finite_real_values(values: Sequence[Real], name: str) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of finite numbers")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a sequence of finite numbers")
        result.append(float(value))
    return result
