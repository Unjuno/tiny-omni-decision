import copy

import numpy as np
import pytest


def test_full_admission_rejects_partial_smoke():
    pytest.importorskip("torch")
    from scripts.lombard_temporal_full import admit_complete_smoke
    from scripts.train_lombard_temporal import ARMS

    pins = {"fixed": "sha"}
    reference = dict(
        status="PASS_FIVE_ARM_TRAIN_ONLY_SMOKE",
        smoke=True,
        gpu_used=False,
        validated_hashes=pins,
        timing_probe={"projected_full_seconds": 100},
        arms={
            a: dict(
                status="PASS_TRAIN_ONLY_SMOKE",
                validation_evaluated=False,
                calibration_evaluated=False,
                updates=2,
                train_unique_assets=4,
                consumed_assets=8,
                head_parameters=106355,
                final_reload_exact=True,
                json_option_permutation_exact=True,
                initial_tensor_sha256="same",
                sampler_sha256="same",
            )
            for a in ARMS
        },
    )
    admit_complete_smoke(reference, pins)
    for key, value in [
        ("updates", 1),
        ("final_reload_exact", False),
        ("status", "RUNNING"),
        ("consumed_assets", 4),
    ]:
        bad = copy.deepcopy(reference)
        bad["arms"][ARMS[0]][key] = value
        with pytest.raises(ValueError):
            admit_complete_smoke(bad, pins)


def test_controls_preserve_donors_and_apply_only_requested_intervention():
    torch = pytest.importorskip("torch")
    from scripts.lombard_temporal_full import controls

    a = torch.arange(3 * 384).reshape(3, 384).float()
    v = torch.arange(3 * 16 * 768).reshape(3, 16, 768).float()
    mask = torch.ones(3, 2)
    changed = controls((a, v, mask), [1, 2, 0])
    torch.testing.assert_close(changed["reverse"][1], v.flip(1), rtol=0, atol=0)
    torch.testing.assert_close(changed["video_mismatch"][0], a, rtol=0, atol=0)
    torch.testing.assert_close(changed["audio_mismatch"][1], v, rtol=0, atol=0)
    assert changed["audio_missing"][2][:, 0].eq(0).all()
    assert changed["video_missing"][1].eq(0).all()
    assert mask.eq(1).all() and a.any() and v.any()
    with pytest.raises(ValueError):
        controls((a, v, mask), [0, 0, 1])


def test_prediction_snapshot_identity_and_probability_checks(tmp_path):
    pytest.importorskip("torch")
    from scripts.lombard_temporal_full import verify_snapshot
    from scripts.train_lombard_temporal import store_prediction

    logits = np.zeros((2, 51), np.float32)
    y = np.zeros((2, 6), np.int64)
    rows = [{"source_id": str(i), "speaker_group": str(i)} for i in range(2)]
    metric = store_prediction(tmp_path, "valid", logits, y, rows)
    actual = verify_snapshot(tmp_path / "valid.npz", rows, y, metric)
    np.testing.assert_array_equal(actual, logits)
    with pytest.raises(ValueError):
        verify_snapshot(tmp_path / "valid.npz", rows[::-1], y, metric)
    with pytest.raises(ValueError, match="freshly replayed"):
        verify_snapshot(
            tmp_path / "valid.npz", rows, y, metric, replayed_logits=np.ones_like(logits)
        )
