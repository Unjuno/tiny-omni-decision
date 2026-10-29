from __future__ import annotations

import hashlib
import json

import pandas as pd

from scripts import prepare_egoviz_vs_sync_selection as selection


def _fixture_rows() -> tuple[pd.DataFrame, set[str], set[str]]:
    groups = [f"group-{index:02}" for index in range(40)]
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256((selection.SPLIT_PREFIX + group).encode()).hexdigest(),
    )
    development = set(ordered[:8])
    training = set(ordered[8:])
    categories = selection.CATEGORIES
    assignment = {group: categories[index // 2] for index, group in enumerate(ordered[:8])}
    for index, group in enumerate(ordered[8:]):
        assignment[group] = categories[index % len(categories)]

    rows = []
    for number, (group, category) in enumerate(assignment.items()):
        rows.append(
            {
                "category": category,
                "group_id": group,
                "mcap_bytes": 1000 + number,
                "chunk_id": f"chunk-{number:02}",
                "sequence_id": f"sequence-{number:02}",
                "mcap_path": f"data/{category}/chunk-{number:02}/chunk-{number:02}.mcap",
                "duration_s": 14.97,
                "n_frames": 450,
            }
        )
    return pd.DataFrame(rows), training, development


def test_selection_is_deterministic_and_group_disjoint(tmp_path, monkeypatch):
    frame, training, development = _fixture_rows()
    index_path = tmp_path / "index.parquet"
    index_path.write_bytes(b"synthetic index")
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "seed": 17,
                "index_sha256": "index-hash",
                "train_group_sha256": sorted(
                    hashlib.sha256(group.encode()).hexdigest() for group in training
                ),
                "development_group_sha256": sorted(
                    hashlib.sha256(group.encode()).hexdigest() for group in development
                ),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(selection, "INDEX_SHA256", "index-hash")
    monkeypatch.setattr(selection, "SPLIT_SHA256", "split-hash")
    monkeypatch.setattr(
        selection, "_sha256", lambda path: "index-hash" if path == index_path else "split-hash"
    )
    monkeypatch.setattr(selection.pd, "read_parquet", lambda _path: frame.copy())

    first = selection.select(index_path, split_path)
    monkeypatch.setattr(
        selection.pd, "read_parquet", lambda _path: frame.sample(frac=1, random_state=9)
    )
    second = selection.select(index_path, split_path)

    assert first == second
    assert first["selection"]["files_total"] == 16
    assert first["selection"]["train_development_group_overlap"] == 0
    assert len(first["selection"]["train_groups"]) == 8
    assert len(first["selection"]["development_groups"]) == 8
    assert set(first["selection"]["train_groups"]).isdisjoint(
        first["selection"]["development_groups"]
    )
