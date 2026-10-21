import pytest

from scripts.bootstrap_av_correspondence import bootstrap


def test_cluster_bootstrap_deterministic_paired_zero():
    rows = [{"quartet_id": str(i // 4), "target_index": i % 4} for i in range(8)]
    condition = {
        "probabilities": [[0.25] * 4 for _ in rows],
        "targets": [r["target_index"] for r in rows],
    }
    conditions = {"best": condition, "untrained": condition}
    result = bootstrap(rows, conditions, draws=20)
    assert result == bootstrap(rows, conditions, draws=20)
    assert result["conditions"]["best"]["accuracy"] == [0.25, 0.25]
    assert all(v == [0, 0] for v in result["paired_best_minus_untrained"].values())
    with pytest.raises(ValueError, match="quartets"):
        bootstrap(rows[:-1], conditions, draws=20)
