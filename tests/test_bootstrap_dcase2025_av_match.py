import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("numpy")

from scripts.bootstrap_dcase2025_av_match import summarize  # noqa: E402


def test_saved_av_match_bootstrap_is_paired_and_never_overwrites(tmp_path):
    run = tmp_path
    targets = [0, 1, 2, 3, 0, 1, 2, 3]
    rows = []
    for index, target in enumerate(targets):
        correct = np.full(4, 0.1)
        correct[target] = 0.7
        rows.append(
            {
                "query_id": f"q{index}",
                "query_parent": f"p{index // 2}",
                "room_id": f"r{index // 4}",
                "target_index": target,
                "probabilities": {
                    "joint": correct.tolist(),
                    "audio_missing": [0.25] * 4,
                    "video_missing": [0.25] * 4,
                    "video_shuffled": [0.25] * 4,
                    "uniform": [0.25] * 4,
                },
            }
        )
    prediction_path = run / "oof-predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (run / "effective-config.json").write_text(
        json.dumps(
            {"outer_protocol": "leave-one-room-out across 9 rooms; every query tested once"}
        ),
        encoding="utf-8",
    )
    (run / "result.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "queries": len(rows),
                "selection_used_outer_test": False,
                "development_test_accessed": False,
                "sealed_audit_accessed": False,
                "predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    result = summarize(run, replicates=64, seed=17)

    assert result["queries"] == 8
    assert result["paired_group_differences"]["room"]["independent_groups"] == 2
    assert result["paired_group_differences"]["parent_recording"]["contrasts"][
        "joint_minus_uniform"
    ]["point"]["accuracy"] == pytest.approx(0.75)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        summarize(run, replicates=64, seed=17)
