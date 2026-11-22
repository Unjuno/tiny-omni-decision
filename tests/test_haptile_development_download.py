from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.pretrained_reuse.download_haptile_validation_calibration import (
    development_tasks,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def manifest_and_counts():
    manifest = json.loads(
        (REPO_ROOT / "manifests/pretrained_reuse/haptile-task-split-v1-20261122.json")
        .read_text(encoding="utf-8")
    )
    counts = {}
    for name in ("validation", "calibration"):
        tasks = manifest["development_splits"][name]
        base, remainder = divmod(manifest["development_episode_counts"][name], len(tasks))
        counts.update({task: base + int(index < remainder)
                       for index, task in enumerate(tasks)})
    return manifest, counts


def test_development_download_only_returns_requested_frozen_task_groups(
    manifest_and_counts,
):
    manifest, counts = manifest_and_counts

    assert development_tasks(manifest, counts, "validation") == sorted(
        manifest["development_splits"]["validation"]
    )
    assert development_tasks(manifest, counts, "calibration") == sorted(
        manifest["development_splits"]["calibration"]
    )


@pytest.mark.parametrize("split_name", ["train", "sealed_audit", "test"])
def test_development_downloader_fails_closed_for_other_splits(
    manifest_and_counts, split_name
):
    manifest, counts = manifest_and_counts
    with pytest.raises(ValueError, match="only frozen validation"):
        development_tasks(manifest, counts, split_name)


def test_development_downloader_rejects_overlapping_public_task_groups(
    manifest_and_counts,
):
    manifest, counts = manifest_and_counts
    modified = copy.deepcopy(manifest)
    modified["development_splits"]["calibration"][0] = (
        modified["development_splits"]["validation"][0]
    )

    with pytest.raises(ValueError, match="overlap"):
        development_tasks(modified, counts, "validation")


def test_development_downloader_checks_task_counts_against_split_commitment(
    manifest_and_counts,
):
    manifest, counts = manifest_and_counts
    counts[manifest["development_splits"]["validation"][0]] += 1

    with pytest.raises(ValueError, match="differ from the split commitment"):
        development_tasks(manifest, counts, "validation")
