"""Fixed full-vocabulary native tasks; no missing-label imputation or difficulty filtering."""

import math

TASKS = {
    "material": ("What material is contacted at the supplied event?", [
        "carpet", "ceramic", "cloth", "dirt", "drywall", "glass", "grass", "gravel", "leaf",
        "metal", "paper", "plastic", "plastic-bag", "rock", "tile", "water", "wood"]),
    "action": ("What action occurs at the supplied contact event?", ["hit", "scratch"]),
    "reaction": ("How does the contacted material react at the supplied event?", [
        "deform", "rigid-motion", "scatter", "splash", "static"]),
}


def tasks_for_parent(parent, split, asset_sha, rows, invalid_windows):
    if invalid_windows:
        raise ValueError("invalid window remains; do not silently shrink the corpus")
    tasks, missing = [], dict.fromkeys(TASKS, 0)
    for index, row in enumerate(rows):
        event = f"greatest-hits/{parent}/{index:04d}"
        for field, (kind, (question, options)) in enumerate(TASKS.items()):
            target = row["fields"][field]
            if target is None:
                missing[kind] += 1
                continue
            if target not in options:
                raise ValueError("unreviewed category; preserve source and fail closed")
            t = row["seconds"]
            tasks.append(dict(id=event + "/" + kind, event_id=event, parent=parent, split=split,
                              date_group=parent[:10], task=kind, question=question,
                              options=list(options), target=options.index(target),
                              asset_sha256=asset_sha, media_file=parent + "_denoised.mp4",
                              event_seconds=t, event_window_seconds=[t - 0.25, t + 0.25],
                              source_line=index + 1, localization="supplied annotation onset",
                              modalities=["audio", "video"]))
    return tasks, missing


def prior_metrics(train_counts, validation_counts):
    if (len(train_counts) != len(validation_counts) or len(train_counts) < 2
            or any(n <= 0 for n in train_counts) or any(n < 0 for n in validation_counts)
            or sum(validation_counts) <= 0):
        raise ValueError("positive TRAIN support and nonempty validation required")
    total = sum(validation_counts)
    probabilities = [n / sum(train_counts) for n in train_counts]
    choice = max(range(len(probabilities)), key=probabilities.__getitem__)
    accuracy = validation_counts[choice] / total
    nll = sum(-n * math.log(p)
              for n, p in zip(validation_counts, probabilities, strict=True)) / total
    squared = sum(p * p for p in probabilities)
    brier = sum(n * (squared - 2 * p + 1)
                for n, p in zip(validation_counts, probabilities, strict=True)) / total
    return dict(count=total, probabilities=probabilities, selected_index=choice, accuracy=accuracy,
                nll=nll, brier=brier, ece=abs(probabilities[choice] - accuracy),
                ece_method="constant confidence; exact occupied-bin absolute gap")
