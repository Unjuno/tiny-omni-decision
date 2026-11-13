"""Pure helpers for the fixed-training modality-gradient diagnostic."""

from itertools import combinations

import torch

MODALITIES = ("text", "image", "audio", "video")


def fixed_update_window(order, selected_step, window_updates):
    if (
        type(selected_step) is not int
        or type(window_updates) is not int
        or selected_step < 1
        or window_updates < 1
        or selected_step > len(order)
        or window_updates > selected_step
    ):
        raise ValueError("valid selected step and nonempty prior update window required")
    start = selected_step - window_updates
    window = order[start:selected_step]
    if len(window) != window_updates or any(
        len(batch) != 8 or any(type(index) is not int or index < 0 for index in batch)
        for batch in window
    ):
        raise ValueError("window must contain complete eight-example updates")
    return start, window


def required_bank_names(observations, observation_ids, query_map):
    if not observation_ids or not query_map:
        raise ValueError("selected TRAIN observations and queries required")
    try:
        names = {
            feature["bank"]
            for ident in observation_ids
            for feature in observations[ident]["features"]
        }
        names.update(entry["bank"] for entry in query_map.values())
    except (KeyError, TypeError) as error:
        raise ValueError("selected TRAIN observation/query bank references required") from error
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("nonempty bank names required")
    return names


def _cosine(left, right):
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if left_norm.item() == 0 or right_norm.item() == 0:
        return None
    return float(torch.dot(left, right).div(left_norm * right_norm).item())


def update_gradient_metrics(modality_gradients, shared_gradients, private_gradients, clip_norm):
    expected = set(MODALITIES)
    if (
        set(modality_gradients) != expected
        or set(shared_gradients) != expected
        or set(private_gradients) != expected
        or type(clip_norm) not in (int, float)
        or clip_norm <= 0
    ):
        raise ValueError("four modality gradients and positive clip norm required")
    groups = (modality_gradients, shared_gradients, private_gradients)
    all_vectors = [value for group in groups for value in group.values()]
    for group in groups:
        vectors = list(group.values())
        reference = vectors[0]
        if (
            not isinstance(reference, torch.Tensor)
            or reference.ndim != 1
            or any(
                not isinstance(value, torch.Tensor)
                or value.ndim != 1
                or value.shape != reference.shape
                or value.device != reference.device
                or not torch.isfinite(value).all()
                for value in vectors
            )
        ):
            raise ValueError("finite, aligned one-dimensional gradient vectors required")
    if any(value.device != all_vectors[0].device for value in all_vectors):
        raise ValueError("gradient scopes must use one device")

    combined = torch.stack([modality_gradients[name] for name in MODALITIES]).mean(0)
    combined_norm = float(torch.linalg.vector_norm(combined).item())
    clipping_scale = min(1.0, float(clip_norm) / combined_norm) if combined_norm else 1.0

    pairwise = {}
    for left, right in combinations(MODALITIES, 2):
        pairwise[f"{left}|{right}"] = {
            "all_parameters": _cosine(modality_gradients[left], modality_gradients[right]),
            "shared_parameters": _cosine(shared_gradients[left], shared_gradients[right]),
        }
    mean_shared_norm = sum(
        float(torch.linalg.vector_norm(shared_gradients[name]).item()) for name in MODALITIES
    ) / len(MODALITIES)
    shared_combined_norm = float(
        torch.linalg.vector_norm(
            torch.stack([shared_gradients[name] for name in MODALITIES]).mean(0)
        ).item()
    )
    return {
        "pairwise_cosine": pairwise,
        "gradient_norms": {
            "all_parameters_by_modality": {
                name: float(torch.linalg.vector_norm(modality_gradients[name]).item())
                for name in MODALITIES
            },
            "shared_parameters_by_modality": {
                name: float(torch.linalg.vector_norm(shared_gradients[name]).item())
                for name in MODALITIES
            },
            "modality_projection_parameters_by_modality": {
                name: float(torch.linalg.vector_norm(private_gradients[name]).item())
                for name in MODALITIES
            },
            "global_batch_before_clip": combined_norm,
        },
        "global_clip_scale": clipping_scale,
        "shared_gradient_cancellation_ratio": (
            shared_combined_norm / mean_shared_norm if mean_shared_norm else 1.0
        ),
    }


def summarize_updates(records):
    if not records:
        raise ValueError("at least one update record required")
    pair_names = sorted(records[0]["pairwise_cosine"])

    def stats(values):
        ordered = sorted(values)
        middle = len(ordered) // 2
        median = (
            ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
        )
        return {
            "mean": sum(ordered) / len(ordered),
            "median": median,
            "p10": ordered[int((len(ordered) - 1) * 0.10)],
            "p90": ordered[int((len(ordered) - 1) * 0.90)],
        }

    pairwise = {}
    for pair in pair_names:
        pairwise[pair] = {}
        for scope in ("all_parameters", "shared_parameters"):
            values = [
                record["pairwise_cosine"][pair][scope]
                for record in records
                if record["pairwise_cosine"][pair][scope] is not None
            ]
            pairwise[pair][scope] = (
                {
                    **stats(values),
                    "negative_fraction": sum(value < 0 for value in values) / len(values),
                    "defined_updates": len(values),
                }
                if values
                else {
                    "mean": None,
                    "median": None,
                    "p10": None,
                    "p90": None,
                    "negative_fraction": None,
                    "defined_updates": 0,
                }
            )
    clip_scales = [record["global_clip_scale"] for record in records]
    return {
        "updates": len(records),
        "pairwise_cosine": pairwise,
        "global_clip_scale": stats(clip_scales),
        "fraction_updates_clipped": sum(value < 1 for value in clip_scales) / len(clip_scales),
        "shared_gradient_cancellation_ratio": stats(
            [record["shared_gradient_cancellation_ratio"] for record in records]
        ),
    }
