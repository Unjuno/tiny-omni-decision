import numpy as np
import pytest


def test_actual_complete_directory_accounting_scope_is_required_not_invented_status():
    from scripts.diagnose_robomnist_lengths import require_parent

    parent = dict(
        status="PASS_COMPLETE_DIRECTORY_GROUP_ACCOUNTING_ONLY",
        archives=60,
        unique_repetition_groups=2135,
        media_members_opened=False,
        training=False,
        sealed_audit_opened=False,
    )
    require_parent(parent)
    with pytest.raises(ValueError):
        require_parent({**parent, "archives": 59})
    with pytest.raises(ValueError):
        require_parent({**parent, "sealed_audit_opened": True})


def test_same_repetition_block_all_classes_is_deterministic_and_pilot_stays_train():
    from scripts.robomnist_length_probe import split_blocks

    a = split_blocks(range(1, 45))
    assert a == split_blocks(reversed(range(1, 45)))
    assert a[1] == "train" and sum(v == "validation" for v in a.values()) == 11
    assert set(a) == set(range(1, 45))
    with pytest.raises(ValueError):
        split_blocks([2, 3])


def test_only_fixed_seven_numeric_lengths_and_no_gt_or_duplicate_stream():
    from scripts.robomnist_length_probe import length_features

    names = [
        f"folder/name_Rx{i}_{kind}.{ext}"
        for kind, ext in [("cam", "mp4"), ("mic", "wav")]
        for i in [1, 2, 3]
    ] + ["folder/name_Rxall_csi.json"]
    sizes = {p: n + 100 for n, p in enumerate(names)}
    a = length_features({"input_members": names}, sizes)
    assert a.shape == (7,) and np.array_equal(a, np.log1p(np.arange(100, 107)))
    with pytest.raises(ValueError):
        length_features({"input_members": names[:-1] + ["folder/name_Rxall_pos.json"]}, sizes)
    with pytest.raises(ValueError):
        length_features({"input_members": names[:-1] + [names[0]]}, sizes)
    with pytest.raises(ValueError):
        length_features({"input_members": names}, {**sizes, names[0]: True})


def test_centroid_training_only_and_missing_class_or_nan_fail_closed():
    from scripts.robomnist_length_probe import centroid_logp

    x = np.array([[0.0], [1.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    p = centroid_logp(x, y, np.array([[0.0], [4.0]]), 2)
    q = centroid_logp(x, y, np.array([[0.0], [4.0], [10000.0]]), 2)
    assert np.array_equal(p, q[:2])
    assert np.array_equal(p.argmax(1), [0, 1])
    assert np.allclose(np.exp(p).sum(1), 1)
    with pytest.raises(ValueError):
        centroid_logp(x, y, x, 3)
    with pytest.raises(ValueError):
        centroid_logp(x, y, np.array([[np.nan]]), 2)


def test_full_class_posteriors_marginalize_without_option_reduction():
    from scripts.robomnist_length_probe import task_logp

    p = np.arange(1, 61, dtype=float)[None, :]
    p /= p.sum(1, keepdims=True)
    for task, mapping, count in [
        ("robot", (np.arange(60) % 20) // 10, 2),
        ("velocity", np.arange(60) // 20, 3),
        ("trajectory", np.arange(60) % 10, 10),
    ]:
        q = np.exp(task_logp(np.log(p), task))
        assert q.shape == (1, count)
        assert np.allclose(q, [[p[0, mapping == i].sum() for i in range(count)]])


def test_unclipped_stable_nll_is_not_replaced_by_probability_floor():
    from scripts.robomnist_length_probe import metrics

    logp = np.array([[0.0, -1000.0]])
    m = metrics(logp, np.array([1]))
    assert m["nll"] == 1000.0 and m["accuracy"] == 0 and m["brier"] == 2
    with pytest.raises(ValueError):
        metrics(np.array([[0.0, 0.0]]), np.array([0]))


def test_paired_bootstrap_keeps_whole_unequal_blocks_and_observed_micro_difference():
    from scripts.robomnist_length_probe import paired_block_interval

    a = np.array([True, False, False, False])
    control = np.zeros(4, bool)
    blocks = np.array([1, 2, 2, 2])
    result = paired_block_interval(a, control, blocks)
    assert result == paired_block_interval(a, control, blocks)
    assert result["delta_accuracy"] == 0.25 and result["repetition_blocks"] == 2
    assert result["interval_95_percentile"] == [0.0, 1.0]


def test_saved_config_rows_and_arrays_are_bound_at_creation_and_read_back(tmp_path):
    from scripts.diagnose_robomnist_lengths import require_products, write, write_arrays

    config, rows = {"policy": "fixed"}, [{"split": "train"}]
    arrays = {"x": np.arange(3, dtype=np.float64)}
    hashes = {
        "config.json": write(tmp_path / "config.json", config),
        "rows.json": write(tmp_path / "rows.json", rows),
        "arrays.npz": write_arrays(tmp_path / "arrays.npz", arrays),
    }
    require_products(tmp_path, hashes, config, rows, arrays)
    (tmp_path / "config.json").write_text('{"policy":"changed"}', encoding="utf8")
    with pytest.raises(ValueError):
        require_products(tmp_path, hashes, config, rows, arrays)


def test_source_or_numeric_failure_saves_incomplete_terminal_and_never_overwrites(
    tmp_path, monkeypatch
):
    import json

    from scripts import diagnose_robomnist_lengths as module

    def fail(output, tick):
        raise RuntimeError("simulated source/hash/numeric guard")

    monkeypatch.setattr(module, "run_body", fail, raising=False)
    output = tmp_path / "fresh"
    assert module.execute(output) is False
    result = json.loads((output / "result.json").read_bytes())
    assert result["status"] == "FAIL_INCOMPLETE_LENGTH_DIAGNOSTIC"
    assert "simulated source" in result["error"]
    original = (output / "result.json").read_bytes()
    with pytest.raises(ValueError):
        module.execute(output)
    assert (output / "result.json").read_bytes() == original


def test_final_readback_resource_growth_cannot_keep_earlier_pass_snapshot(tmp_path, monkeypatch):
    import json

    from scripts import diagnose_robomnist_lengths as module

    monkeypatch.setattr(
        module,
        "run_body",
        lambda output, tick: dict(
            status="PASS_FIXED_DIRECTORY_LENGTH_SHORTCUT_DIAGNOSTIC_ONLY",
            elapsed_seconds=1.0,
            peak_cpu_bytes=100,
        ),
    )
    ticks = iter([0.0, 121.0, 121.0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))
    output = tmp_path / "fresh"
    assert module.execute(output) is False
    result = json.loads((output / "result.json").read_bytes())
    assert result["status"] == "FAIL_INCOMPLETE_LENGTH_DIAGNOSTIC"
    assert result["elapsed_seconds"] == 121.0
