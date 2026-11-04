"""Leakage-aware grouped folds for TRAIN-only QASC diagnostics."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

from .qasc import _normalized


def _row_identities(row: dict[str, Any]) -> tuple[str, str, str, int]:
    source_record_id = row.get("source_record_id")
    question = row.get("question")
    options = row.get("options")
    target = row.get("target")
    if not isinstance(source_record_id, str) or ":" not in source_record_id:
        raise ValueError("QASC row needs a fact-pair group in source_record_id")
    group_id, row_id = source_record_id.split(":", 1)
    if not group_id or not row_id or not isinstance(question, str) or not question.strip():
        raise ValueError("QASC row identity/question is malformed")
    if (
        not isinstance(options, list)
        or len(options) != 8
        or any(not isinstance(x, str) for x in options)
    ):
        raise ValueError("QASC rows require eight string options")
    if not isinstance(target, str) or target not in options:
        raise ValueError("QASC target must match one supplied option")
    normalized_question = _normalized(question)
    normalized_options = sorted(_normalized(option) for option in options)
    content = json.dumps(
        {"question": normalized_question, "options": normalized_options},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        group_id,
        normalized_question,
        hashlib.sha256(content.encode("utf-8")).hexdigest(),
        options.index(target),
    )


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _connected_components(
    rows: list[dict[str, Any]],
) -> tuple[list[tuple[str, str, str, int]], _UnionFind, list[list[int]]]:
    identities = [_row_identities(row) for row in rows]
    row_ids = [row.get("id") for row in rows]
    if any(not isinstance(row_id, str) or not row_id for row_id in row_ids):
        raise ValueError("QASC rows require nonempty string IDs")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("QASC row IDs must be unique")
    union_find = _UnionFind(len(rows))
    first_by_identity: dict[tuple[str, str], int] = {}
    for index, (fact_group, question, content, _position) in enumerate(identities):
        for key in (("fact", fact_group), ("question", question), ("content", content)):
            if key in first_by_identity:
                union_find.union(index, first_by_identity[key])
            else:
                first_by_identity[key] = index
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[union_find.find(index)].append(index)
    return identities, union_find, list(components.values())


def qasc_train_component_ids(rows: list[dict[str, Any]]) -> list[str]:
    """Return stable cluster IDs for the same fact/question/content graph as CV."""
    _identities, _union_find, components = _connected_components(rows)
    component_ids = [""] * len(rows)
    for component in components:
        payload = "\n".join(sorted(str(rows[index]["id"]) for index in component))
        component_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        for index in component:
            component_ids[index] = component_id
    return component_ids


def build_qasc_train_folds(
    rows: list[dict[str, Any]], *, folds: int = 5, seed: int = 17
) -> tuple[list[int], dict[str, Any]]:
    """Assign every row to one deterministic fold without fact/question/content leakage.

    Connected components join rows sharing an unordered fact-pair group, a normalized
    question, or normalized question+choice content. A greedy size/answer-position
    balance is used only to distribute whole components; no model output is consulted.
    """
    if folds < 2 or len(rows) < folds:
        raise ValueError("at least two folds and one row per fold are required")
    identities, union_find, component_rows = _connected_components(rows)
    if len(component_rows) < folds:
        raise ValueError("not enough independent connected components for requested folds")

    total_positions = Counter(position for _fact, _question, _content, position in identities)
    target_size = len(rows) / folds
    target_positions = {position: total_positions[position] / folds for position in range(8)}
    tie_keys = {
        id(group): hashlib.sha256(
            (f"{seed}:" + "\n".join(sorted(str(rows[index]["id"]) for index in group))).encode(
                "utf-8"
            )
        ).hexdigest()
        for group in component_rows
    }
    component_rows.sort(key=lambda group: (-len(group), tie_keys[id(group)]))
    fold_indices = [[] for _ in range(folds)]
    fold_positions = [Counter() for _ in range(folds)]
    for group in component_rows:
        group_positions = Counter(identities[index][3] for index in group)
        candidate_scores = []
        empty_folds = [index for index, assigned in enumerate(fold_indices) if not assigned]
        if empty_folds:
            selected = min(
                empty_folds,
                key=lambda index: hashlib.sha256(
                    f"{seed}:{tie_keys[id(group)]}:{index}".encode()
                ).hexdigest(),
            )
        else:
            for fold_index in range(folds):
                score = 0.0
                for candidate_fold in range(folds):
                    size_after = len(fold_indices[candidate_fold])
                    if candidate_fold == fold_index:
                        size_after += len(group)
                    score += ((size_after - target_size) / target_size) ** 2
                    for position in range(8):
                        count_after = fold_positions[candidate_fold][position]
                        if candidate_fold == fold_index:
                            count_after += group_positions[position]
                        score += (
                            (count_after - target_positions[position])
                            / max(target_positions[position], 1.0)
                        ) ** 2 / 8
                tie = hashlib.sha256(
                    f"{seed}:{tie_keys[id(group)]}:{fold_index}".encode()
                ).hexdigest()
                candidate_scores.append((score, len(fold_indices[fold_index]), tie, fold_index))
            selected = min(candidate_scores)[-1]
        fold_indices[selected].extend(group)
        fold_positions[selected].update(group_positions)

    assignment = [-1] * len(rows)
    fold_report = []
    for fold_index, indices in enumerate(fold_indices):
        if not indices:
            raise ValueError("fold assignment produced an empty fold")
        for index in indices:
            assignment[index] = fold_index
        fold_report.append(
            {
                "fold": fold_index,
                "rows": len(indices),
                "components": len({union_find.find(index) for index in indices}),
                "answer_position_counts": {str(i): fold_positions[fold_index][i] for i in range(8)},
                "ids_sha256": hashlib.sha256(
                    "\n".join(str(rows[i]["id"]) for i in indices).encode("utf-8")
                ).hexdigest(),
            }
        )
    if any(value < 0 for value in assignment):
        raise AssertionError("not all QASC rows were assigned to a fold")

    component_fold = {}
    for index, fold_index in enumerate(assignment):
        root = union_find.find(index)
        previous = component_fold.setdefault(root, fold_index)
        if previous != fold_index:
            raise AssertionError("connected component was split across folds")
    report = {
        "rows": len(rows),
        "folds": folds,
        "seed": seed,
        "connected_components": len(component_rows),
        "fact_pair_groups": len({item[0] for item in identities}),
        "normalized_questions": len({item[1] for item in identities}),
        "normalized_question_choice_contents": len({item[2] for item in identities}),
        "folds_report": fold_report,
        "all_rows_assigned_once": len(assignment) == len(rows)
        and all(0 <= x < folds for x in assignment),
        "group_integrity_verified": True,
    }
    return assignment, report


def audit_qasc_train_folds(
    rows: list[dict[str, Any]], assignment: list[int], *, folds: int = 5
) -> list[dict[str, int]]:
    """Prove fact-pair, normalized-question, and normalized-content separation per fold."""
    if len(rows) != len(assignment) or any(not 0 <= fold < folds for fold in assignment):
        raise ValueError("fold assignment must cover each row exactly once")
    identities = [_row_identities(row) for row in rows]
    audits = []
    for held_fold in range(folds):
        train = [i for i, value in enumerate(assignment) if value != held_fold]
        held = [i for i, value in enumerate(assignment) if value == held_fold]
        if not train or not held:
            raise ValueError("each fold must have nonempty train and held-out rows")
        overlap_counts = []
        for column in range(3):
            train_ids = {identities[i][column] for i in train}
            held_ids = {identities[i][column] for i in held}
            overlap_counts.append(len(train_ids & held_ids))
        if any(overlap_counts):
            raise ValueError(
                "QASC fold leakage: "
                f"{overlap_counts[0]} fact-pair groups, "
                f"{overlap_counts[1]} normalized questions, "
                f"{overlap_counts[2]} normalized question-choice contents"
            )
        audits.append(
            {
                "fold": held_fold,
                "train_rows": len(train),
                "held_rows": len(held),
                "train_held_fact_pair_overlap": 0,
                "train_held_question_overlap": 0,
                "train_held_content_overlap": 0,
            }
        )
    return audits
