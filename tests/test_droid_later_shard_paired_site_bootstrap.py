from __future__ import annotations

import pytest

from scripts.audit_droid_later_shard_paired_site_bootstrap import (
    paired_site_cluster_bootstrap,
)


def _row(row_id: str, group_id: str, site: str, target: str) -> dict[str, object]:
    probs = [1 / 7] * 7
    return {
        "id": row_id,
        "group_id": group_id,
        "site": site,
        "target_option_id": target,
        "probabilities": {"sensor": probs, "video_sensor": probs},
    }


def test_site_bootstrap_is_paired_and_deterministic() -> None:
    rows = [
        _row("a", "ga", "site-a", "stop"),
        _row("b", "gb", "site-a", "x+"),
        _row("c", "gc", "site-b", "y-"),
        _row("d", "gd", "site-b", "z+")
    ]

    first = paired_site_cluster_bootstrap(rows, repetitions=100, seed=17)
    second = paired_site_cluster_bootstrap(rows, repetitions=100, seed=17)

    assert first == second
    assert first["point_delta_video_sensor_minus_sensor"] == {
        "accuracy": 0.0,
        "nll": 0.0,
        "brier": 0.0,
    }
    assert first["site_count"] == 2


def test_site_bootstrap_rejects_duplicate_group_ids() -> None:
    rows = [_row("a", "same", "site-a", "stop"), _row("b", "same", "site-b", "x+")]

    with pytest.raises(ValueError, match="unique independent episode group"):
        paired_site_cluster_bootstrap(rows, repetitions=10)


def test_site_bootstrap_requires_multiple_sites() -> None:
    rows = [_row("a", "ga", "site-a", "stop"), _row("b", "gb", "site-a", "x+")]

    with pytest.raises(ValueError, match="at least two collection sites"):
        paired_site_cluster_bootstrap(rows, repetitions=10)
