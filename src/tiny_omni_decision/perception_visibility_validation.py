"""Fixed, one-time transfer diagnostics for frozen Perception visibility heads."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import numpy as np

from tiny_omni_decision.perception_visibility_evaluation import score_binary_logits


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_validation_manifest(
    selection: Mapping[str, object],
    media_validation: Mapping[str, object],
    *,
    selection_sha256: str,
    media_validation_sha256: str,
) -> dict[str, object]:
    """Create fixed two-second AV windows for the previously selected valid cohort.

    Windows are centered on each annotated event when possible. If clip boundaries
    prevent a centered window, the window is shifted only enough to remain inside
    both decoded streams; no example is dropped or reselected.
    """
    if selection.get("status") != "PREFLIGHT_ONLY_NOT_EVALUATED":
        raise ValueError("selection must remain the frozen preflight cohort")
    if selection.get("source_split") != "official validation only":
        raise ValueError("official validation cohort required")
    if selection.get("official_test_split_opened") is not False:
        raise ValueError("official test must remain unopened")
    if media_validation.get("status") != "PASS_MEDIA_HASH_AND_DECODE_METADATA":
        raise ValueError("verified media receipt required")
    if media_validation.get("selection_sha256") != selection_sha256:
        raise ValueError("media receipt belongs to another selection")
    examples = selection.get("events")
    media_examples = media_validation.get("examples_manifest")
    if not isinstance(examples, list) or not isinstance(media_examples, list):
        raise ValueError("selection and media receipt must contain event records")
    if len(examples) != 200 or len(media_examples) != 200:
        raise ValueError("the fixed cohort must contain exactly 200 events")

    media_by_video = {row["video_id"]: row for row in media_examples}
    if len(media_by_video) != len(media_examples):
        raise ValueError("duplicate media video identity")
    if {row.get("video_id") for row in examples} != set(media_by_video):
        raise ValueError("media identities differ from the frozen cohort")
    targets = [row.get("target") for row in examples]
    if targets.count(0) != 100 or targets.count(1) != 100:
        raise ValueError("the fixed cohort must remain balanced 100/class")

    rows: list[dict[str, object]] = []
    shifted: list[str] = []
    seen: set[str] = set()
    for event in examples:
        video_id = event.get("video_id")
        event_id = event.get("event_id")
        target = event.get("target")
        if not isinstance(video_id, str) or not video_id or video_id in seen:
            raise ValueError("one unique video per event is required")
        seen.add(video_id)
        if type(event_id) is not int or type(target) is not int or target not in (0, 1):
            raise ValueError("event identity and binary target are required")
        media = media_by_video[video_id]
        if media.get("target") != target:
            raise ValueError("media target differs from selection")
        if media.get("event_start_s") != event.get("start_event_s"):
            raise ValueError("media event start differs from selection")
        if media.get("event_end_s") != event.get("end_event_s"):
            raise ValueError("media event end differs from selection")
        audio_duration = float(media["audio_duration_s"])
        video_duration = float(media["video_duration_s"])
        available = min(audio_duration, video_duration)
        if not math.isfinite(available) or available < 2.0:
            raise ValueError(f"two-second window unavailable for {video_id}")
        start = float(event["start_event_s"])
        end = float(event["end_event_s"])
        center = (start + end) / 2.0
        desired_window_start = center - 1.0
        actual_window_start = min(max(desired_window_start, 0.0), available - 2.0)
        window = [actual_window_start, actual_window_start + 2.0]
        if abs(actual_window_start - desired_window_start) > 1e-7:
            shifted.append(video_id)
        frame_times = [window[0] + index * 2.0 / 7.0 for index in range(8)]
        rows.append(
            {
                "sample_id": f"{video_id}__visibility_{event_id}",
                "split": "validation",
                "video_id": video_id,
                "video_event_id": event_id,
                "video_window_s": window,
                "video_frame_times_s": frame_times,
                "target": target,
                "target_index": 0,
                "options": [{"event_id": event_id, "role": "target", "audio_window_s": window}],
            }
        )
    return {
        "train": [],
        "validation": rows,
        "metadata": {
            "scope": "one-time fixed-readout transfer diagnostic on official validation",
            "source_split": "official validation",
            "selection_sha256": selection_sha256,
            "media_validation_sha256": media_validation_sha256,
            "feature_window_seconds": 2.0,
            "video_frame_count": 8,
            "frame_sampling": "uniform inclusive endpoints; boundary-clamped to decoded media",
            "boundary_shift_count": len(shifted),
            "boundary_shift_video_ids": shifted,
            "labels_used_for_selection": True,
            "validation_is_blind": False,
            "training_or_checkpoint_selection": False,
            "official_test_or_sealed_audit_opened": False,
        },
    }


def paired_metric_bootstrap(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int = 17,
    repetitions: int = 10_000,
) -> dict[str, object]:
    """Paired video-level bootstrap for B minus A on the fixed one-event/video cohort."""
    a = np.asarray(logits_a, dtype=np.float64).reshape(-1)
    b = np.asarray(logits_b, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    if a.size == 0 or b.size != a.size or y.size != a.size:
        raise ValueError("paired logits and targets must be nonempty matching vectors")
    if type(seed) is not int or seed < 0 or type(repetitions) is not int or repetitions < 1:
        raise ValueError("invalid bootstrap seed or repetition count")
    scored_a = score_binary_logits(a, y)
    scored_b = score_binary_logits(b, y)
    acc_a = (np.asarray(scored_a["predictions"]) == y).astype(np.float64)
    acc_b = (np.asarray(scored_b["predictions"]) == y).astype(np.float64)
    p_a = np.asarray(scored_a["probability_visible"], dtype=np.float64)
    p_b = np.asarray(scored_b["probability_visible"], dtype=np.float64)
    nll_a = np.logaddexp(0.0, a) - y * a
    nll_b = np.logaddexp(0.0, b) - y * b
    brier_a = np.square(p_a - y)
    brier_b = np.square(p_b - y)
    point = {
        "accuracy": float(np.mean(acc_b - acc_a)),
        "nll": float(np.mean(nll_b - nll_a)),
        "brier": float(np.mean(brier_b - brier_a)),
    }
    values = np.empty((repetitions, 3), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for i in range(repetitions):
        sample = rng.integers(0, a.size, size=a.size)
        values[i] = (
            np.mean(acc_b[sample] - acc_a[sample]),
            np.mean(nll_b[sample] - nll_a[sample]),
            np.mean(brier_b[sample] - brier_a[sample]),
        )
    return {
        "delta_b_minus_a": point,
        "interval_95_percentile": {
            name: [float(x) for x in np.quantile(values[:, index], [0.025, 0.975])]
            for index, name in enumerate(("accuracy", "nll", "brier"))
        },
        "bootstrap_repetitions": repetitions,
        "seed": seed,
        "interval_method": "paired video-level percentile bootstrap",
    }
