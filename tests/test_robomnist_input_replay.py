import pytest


def run():
    from scripts.fetch_robomnist_input_smoke import GROUP

    return dict(
        status="PASS_FIXED_NATIVE_INPUT_MECHANICS_ONLY",
        raw_files=[dict(source_member=f"input-{i}") for i in range(7)],
        config=dict(group=dict(group_id=GROUP)),
        models_loaded=False,
        training=False,
        position_gt_opened=False,
        sealed_audit_opened=False,
        gpu_used=False,
        quality_measured=False,
        source_pins_unchanged=True,
        elapsed_seconds=200,
        reserved_transfer_bytes=1000,
        actual_transfer_bytes=900,
        peak_cpu_bytes=100,
        sync_assessment="UNVERIFIED_COMMON_CLOCK: native clocks",
        array_bank=dict(alignment_only_keys=[f"module_{i}_timestamps_ns" for i in [1, 2, 3]]),
    )


def test_mechanics_success_does_not_launder_sync_or_partial_inputs():
    from scripts.replay_robomnist_input import require_run

    a = run()
    require_run(a)
    a["sync_assessment"] = "SYNC_PASS"
    with pytest.raises(ValueError):
        require_run(a)
    a = run()
    a["raw_files"].pop()
    with pytest.raises(ValueError):
        require_run(a)


def test_transfer_overstatement_omitted_raw_bindings_or_gt_are_rejected():
    from scripts.replay_robomnist_input import require_run

    a = run()
    a["actual_transfer_bytes"] = 1001
    with pytest.raises(ValueError):
        require_run(a)
    a = run()
    a["position_gt_opened"] = True
    with pytest.raises(ValueError):
        require_run(a)
    a = run()
    a["array_bank"]["alignment_only_keys"] = []
    with pytest.raises(ValueError):
        require_run(a)


def test_every_native_array_must_match_shape_dtype_and_bytes():
    import numpy as np

    from scripts.replay_robomnist_input import compare_bank

    expected = {"camera_1_rgb_samples": np.zeros((3, 2, 2, 3), np.uint8)}
    compare_bank(expected, expected)
    bad = {"camera_1_rgb_samples": expected["camera_1_rgb_samples"].astype(np.float32)}
    with pytest.raises(ValueError):
        compare_bank(expected, bad)
    with pytest.raises(ValueError):
        compare_bank(expected, {})


def test_new_decoder_library_and_source_pins_are_mandatory_not_only_parent_pins():
    from scripts.replay_robomnist_input import require_pins

    required = {"old-parent": "a" * 64, "decoder-library": "b" * 64, "new-helper": "c" * 64}
    require_pins(required, required)
    missing = {"old-parent": required["old-parent"]}
    with pytest.raises(ValueError):
        require_pins(required, missing)
    changed = {**required, "new-helper": "d" * 64}
    with pytest.raises(ValueError):
        require_pins(required, changed)
