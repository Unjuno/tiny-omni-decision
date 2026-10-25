"""Deterministic four-choice examples from explicit Open Images labels."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator, Mapping, Set
from typing import Any

MINIMUM_OPENIMAGES_TRAIN_SUPPORT = 1000
EXCLUDED_OPENIMAGES_NAME_TERMS = frozenset(
    {
        "arm",
        "baby",
        "body",
        "boy",
        "child",
        "ear",
        "eye",
        "face",
        "finger",
        "foot",
        "girl",
        "hair",
        "hand",
        "head",
        "human",
        "leg",
        "man",
        "mouth",
        "neck",
        "nose",
        "person",
        "skin",
        "teeth",
        "tooth",
        "woman",
    }
)


def _rank(seed: int, image_id: str, purpose: str, value: str) -> bytes:
    payload = f"openimages-v6-four-choice-v1|{seed}|{image_id}|{purpose}|{value}"
    return hashlib.sha256(payload.encode("utf-8")).digest()


def build_sibling_groups(
    hierarchy: Mapping[str, Any], *, minimum_alternatives: int = 3
) -> dict[str, tuple[str, ...]]:
    """Map each label to siblings under its deepest sufficiently broad parent."""
    if minimum_alternatives < 1:
        raise ValueError("minimum_alternatives must be positive")
    best_groups: dict[str, tuple[int, tuple[str, ...]]] = {}

    def visit(node: Mapping[str, Any], depth: int, ancestors: frozenset[str]) -> None:
        label = node.get("LabelName")
        if not isinstance(label, str) or not label:
            raise ValueError("every hierarchy node needs a nonempty LabelName")
        if label in ancestors:
            raise ValueError(f"cycle in hierarchy at label: {label}")

        children = node.get("Subcategory", [])
        if not isinstance(children, list):
            raise ValueError(f"Subcategory must be a list for {label}")
        child_labels = []
        for child in children:
            if not isinstance(child, Mapping):
                raise ValueError(f"invalid hierarchy child under {label}")
            child_label = child.get("LabelName")
            if not isinstance(child_label, str) or not child_label:
                raise ValueError(f"every hierarchy child under {label} needs a LabelName")
            child_labels.append(child_label)
        if len(child_labels) != len(set(child_labels)):
            raise ValueError(f"duplicate child label under hierarchy parent {label}")

        if len(child_labels) - 1 >= minimum_alternatives:
            siblings = tuple(sorted(child_labels))
            for child_label in child_labels:
                candidate = (depth + 1, tuple(item for item in siblings if item != child_label))
                previous = best_groups.get(child_label)
                if previous is None or candidate[0] > previous[0] or (
                    candidate[0] == previous[0] and candidate[1] < previous[1]
                ):
                    best_groups[child_label] = candidate

        for child in children:
            visit(child, depth + 1, ancestors | {label})

    visit(hierarchy, 0, frozenset())
    return {label: group for label, (_, group) in sorted(best_groups.items())}


def make_four_choice_example(
    *,
    image_id: str,
    positive_labels: Set[str],
    explicit_negative_labels: Set[str],
    sibling_groups: Mapping[str, tuple[str, ...]],
    target_scope: Set[str],
    display_names: Mapping[str, str] | None = None,
    seed: int,
) -> dict[str, Any] | None:
    """Build one image-level four-choice task, using explicit negatives only.

    Returns ``None`` when no positive target has three explicitly negative
    siblings. Unmentioned labels are never treated as negative.
    """
    if not isinstance(image_id, str) or not image_id.strip():
        raise ValueError("image_id must be a nonempty stable identity")
    if not target_scope:
        raise ValueError("target_scope must be nonempty and frozen from training support")
    overlap = set(positive_labels) & set(explicit_negative_labels)
    if overlap:
        raise ValueError("labels are both positive and negative for one image")

    eligible: list[tuple[str, tuple[str, ...]]] = []
    for target in sorted(set(positive_labels) & set(target_scope)):
        negative_siblings = sorted(
            set(sibling_groups.get(target, ())) & set(explicit_negative_labels)
        )
        if len(negative_siblings) >= 3:
            chosen = tuple(
                sorted(
                    negative_siblings,
                    key=lambda label: _rank(seed, image_id, f"negative:{target}", label),
                )[:3]
            )
            eligible.append((target, chosen))
    if not eligible:
        return None

    target, negatives = min(
        eligible,
        key=lambda item: _rank(seed, image_id, "target", item[0]),
    )
    option_labels = [target, *negatives]
    if display_names is None:
        raise ValueError("display_names are required to build user-facing options")
    names: dict[str, str] = {}
    for label in option_labels:
        name = display_names.get(label)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"missing display name for option label {label}")
        names[label] = name.strip()
    if len(set(names.values())) != len(names):
        raise ValueError("selected option labels have duplicate display names")
    option_labels.sort(key=lambda label: _rank(seed, image_id, "option-order", label))
    options = [names[label] for label in option_labels]
    return {
        "sample_id": f"openimages-v6:{image_id}",
        "image_id": image_id,
        "question": "Which object is present in the image?",
        "options": options,
        "target": names[target],
        "target_position": option_labels.index(target),
        "target_label": target,
        "negative_labels": list(negatives),
    }


def select_target_classes(
    train_support_counts: Mapping[str, int],
    display_names: Mapping[str, str],
    *,
    minimum_train_support: int,
    excluded_name_terms: Set[str],
) -> dict[str, int]:
    """Freeze eligible targets using training support and a predeclared name scope.

    Counts are expected to be unique training image counts with one positive
    target and three explicit negative siblings. Validation data is intentionally
    not accepted by this function.
    """
    if minimum_train_support < 1:
        raise ValueError("minimum_train_support must be positive")
    excluded = {term.casefold() for term in excluded_name_terms}
    selected: dict[str, int] = {}
    for label, count in sorted(train_support_counts.items()):
        if not isinstance(label, str) or not label:
            raise ValueError("training support keys must be nonempty class labels")
        if not isinstance(count, int) or count < 0:
            raise ValueError("training support counts must be nonnegative integers")
        display_name = display_names.get(label)
        if not isinstance(display_name, str) or not display_name.strip():
            raise ValueError(f"missing display name for {label}")
        tokens = set(re.findall(r"[a-z0-9]+", display_name.casefold()))
        if count >= minimum_train_support and not tokens.intersection(excluded):
            selected[label] = count
    return selected


def group_verification_labels(
    rows: Iterable[Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Stream Source=verification labels by ImageID, requiring sorted input."""
    current_id: str | None = None
    positives: set[str] = set()
    negatives: set[str] = set()
    last_id: str | None = None

    def record() -> dict[str, Any]:
        conflict = positives & negatives
        if conflict:
            raise ValueError(f"labels are both positive and negative for image {current_id}")
        return {
            "image_id": current_id,
            "positive_labels": set(positives),
            "negative_labels": set(negatives),
        }

    for row in rows:
        image_id = row.get("ImageID")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("annotation row is missing ImageID")
        if last_id is not None and image_id < last_id:
            raise ValueError("annotation rows must be sorted by ImageID")
        last_id = image_id
        if image_id != current_id:
            if current_id is not None:
                yield record()
            current_id = image_id
            positives.clear()
            negatives.clear()

        if row.get("Source") != "verification":
            continue
        label = row.get("LabelName")
        if not isinstance(label, str) or not label:
            raise ValueError("verification annotation row is missing LabelName")
        try:
            confidence = float(row.get("Confidence"))
        except (TypeError, ValueError) as exc:
            raise ValueError("verification confidence must be numeric") from exc
        if confidence == 1.0:
            positives.add(label)
        elif confidence == 0.0:
            negatives.add(label)
        else:
            raise ValueError("verification confidence must be explicitly positive or negative")
    if current_id is not None:
        yield record()


def count_supported_targets(
    grouped_labels: Iterable[Mapping[str, Any]],
    *,
    sibling_groups: Mapping[str, tuple[str, ...]],
    allowed_image_ids: Set[str] | None = None,
) -> dict[str, int]:
    """Count unique images with positive targets and three explicit negatives."""
    counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    for row in grouped_labels:
        image_id = row.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("grouped labels are missing image_id")
        if image_id in seen_ids:
            raise ValueError(f"duplicate grouped image identity: {image_id}")
        seen_ids.add(image_id)
        if allowed_image_ids is not None and image_id not in allowed_image_ids:
            continue
        positives = row.get("positive_labels")
        negatives = row.get("negative_labels")
        if not isinstance(positives, set) or not isinstance(negatives, set):
            raise ValueError("grouped labels need positive_labels and negative_labels sets")
        if positives & negatives:
            raise ValueError(f"labels conflict for image {image_id}")
        for target in positives:
            if len(set(sibling_groups.get(target, ())) & negatives) >= 3:
                counts[target] = counts.get(target, 0) + 1
    return dict(sorted(counts.items()))


def select_balanced_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    target_labels: tuple[str, ...],
    per_target_limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a deterministic, class-balanced set with no repeated image IDs."""
    if per_target_limit < 1:
        raise ValueError("per_target_limit must be positive")
    if not target_labels or len(set(target_labels)) != len(target_labels):
        raise ValueError("target_labels must be a nonempty unique sequence")
    targets = tuple(sorted(target_labels))
    by_target: dict[str, list[dict[str, Any]]] = {target: [] for target in targets}
    seen_pairs: set[tuple[str, str]] = set()
    for candidate in candidates:
        image_id = candidate.get("image_id")
        target = candidate.get("target_label")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("candidate is missing image_id")
        if target not in by_target:
            raise ValueError("candidate target is outside the frozen target scope")
        pair = (image_id, target)
        if pair in seen_pairs:
            raise ValueError("duplicate image/target candidate")
        seen_pairs.add(pair)
        by_target[target].append(dict(candidate))

    for target, options in by_target.items():
        options.sort(
            key=lambda item: (
                _rank(seed, item["image_id"], f"balanced:{target}", item["image_id"]),
                item["image_id"],
            )
        )

    positions = {target: 0 for target in targets}
    counts = {target: 0 for target in targets}
    used_images: set[str] = set()
    selected: list[dict[str, Any]] = []
    while True:
        progressed = False
        for target in targets:
            if counts[target] >= per_target_limit:
                continue
            options = by_target[target]
            position = positions[target]
            while position < len(options) and options[position]["image_id"] in used_images:
                position += 1
            positions[target] = position
            if position >= len(options):
                continue
            candidate = options[position]
            positions[target] += 1
            used_images.add(candidate["image_id"])
            selected.append(candidate)
            counts[target] += 1
            progressed = True
        if not progressed:
            return selected


def iter_target_candidates(
    grouped_labels: Iterable[Mapping[str, Any]],
    *,
    target_labels: tuple[str, ...],
    sibling_groups: Mapping[str, tuple[str, ...]],
) -> Iterator[dict[str, Any]]:
    """Yield one candidate per supported image/target pair in frozen scope."""
    if not target_labels or len(set(target_labels)) != len(target_labels):
        raise ValueError("target_labels must be a nonempty unique sequence")
    target_scope = set(target_labels)
    seen_ids: set[str] = set()
    for row in grouped_labels:
        image_id = row.get("image_id")
        positives = row.get("positive_labels")
        negatives = row.get("negative_labels")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("grouped labels are missing image_id")
        if image_id in seen_ids:
            raise ValueError(f"duplicate grouped image identity: {image_id}")
        seen_ids.add(image_id)
        if not isinstance(positives, set) or not isinstance(negatives, set):
            raise ValueError("grouped labels need positive_labels and negative_labels sets")
        if positives & negatives:
            raise ValueError(f"labels conflict for image {image_id}")
        for target in sorted(positives & target_scope):
            if len(set(sibling_groups.get(target, ())) & negatives) >= 3:
                yield {
                    "image_id": image_id,
                    "target_label": target,
                    "positive_labels": set(positives),
                    "negative_labels": set(negatives),
                }


def build_four_choice_examples(
    grouped_labels: Iterable[Mapping[str, Any]],
    *,
    target_labels: tuple[str, ...],
    sibling_groups: Mapping[str, tuple[str, ...]],
    display_names: Mapping[str, str],
    per_target_limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build balanced, image-unique examples for a frozen target class list."""
    candidates = list(
        iter_target_candidates(
            grouped_labels,
            target_labels=target_labels,
            sibling_groups=sibling_groups,
        )
    )
    selected = select_balanced_candidates(
        candidates,
        target_labels=target_labels,
        per_target_limit=per_target_limit,
        seed=seed,
    )
    examples = []
    for candidate in selected:
        target = candidate["target_label"]
        example = make_four_choice_example(
            image_id=candidate["image_id"],
            positive_labels={target},
            explicit_negative_labels=candidate["negative_labels"],
            sibling_groups=sibling_groups,
            target_scope=set(target_labels),
            display_names=display_names,
            seed=seed,
        )
        if example is None or example["target_label"] != target:
            raise RuntimeError("selected candidate could not be built as a four-choice example")
        examples.append(example)
    return examples
