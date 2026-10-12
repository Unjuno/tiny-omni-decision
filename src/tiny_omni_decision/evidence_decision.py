"""Explicit evidence gate for fixed-taxonomy inference; no confidence threshold."""

from tiny_omni_decision.classification_json import classification_json

MODALITIES = frozenset(("text", "audio", "image", "screen", "video", "sensor"))


def evidence_gated_json(predict, class_ids, options, *, required, available):
    """Caller supplies validated availability, including task-specific freshness.

    This gate does not inspect payloads or establish clock alignment. False means
    unavailable evidence, not a numerical zero. Required modalities belong to
    the fixed model contract, never to a validation-tuned runtime choice.
    """
    required = tuple(required)
    if not required or len(set(required)) != len(required) or not set(required) <= MODALITIES:
        raise ValueError("unique supported required modalities needed")
    if not set(available) <= MODALITIES or any(type(v) is not bool for v in available.values()):
        raise ValueError("availability must use explicit boolean modality flags")
    classes = list(class_ids)
    # Validate the option contract even when inference will be skipped.
    classification_json([1 / len(classes)] * len(classes) if classes else [], classes, options)
    missing = sorted(m for m in required if not available.get(m, False))
    if missing:
        return {"schema_version": "1.0", "status": "abstain",
                "selected_option_id": None, "reason": "insufficient_evidence",
                "missing_modalities": missing,
                "options": [{"option_id": o["option_id"], "probability": None}
                            for o in options]}
    return classification_json(predict(), classes, options)


def aligned_evidence_json(predict, batch, class_ids, options, *, required_sources,
                          query_time_ms, max_age_ms):
    """Gate a time-indexed scorer bound to exact source IDs.

    predict receives only the selected source indices/ages. Its adapter must
    verify/load corresponding payloads. Episode-wide pooled features are not
    causal samples and must not be passed through this interface as if they were.
    """
    from tiny_omni_decision.observation_alignment import align_observations

    if not required_sources or len(set(required_sources.values())) != len(required_sources):
        raise ValueError("distinct required source IDs needed")
    if [o.option_id for o in batch.options] != [o["option_id"] for o in options]:
        raise ValueError("output options must match observation batch IDs and order")
    aligned = align_observations(batch, [query_time_ms], max_age_ms=max_age_ms)
    selected, available = {}, {}
    for modality, source_id in required_sources.items():
        stream = aligned.get(source_id)
        if stream is not None and stream["modality"] != modality:
            raise ValueError("required source modality mismatch")
        available[modality] = stream is not None and stream["available"][0]
        if available[modality]:
            selected[source_id] = {"index": stream["indices"][0],
                                   "age_ms": stream["age_ms"][0]}
    return evidence_gated_json(lambda: predict(selected), class_ids, options,
                               required=required_sources, available=available)
