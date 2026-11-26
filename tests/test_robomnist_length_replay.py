import numpy as np
import pytest


def test_metadata_diagnostic_cannot_be_presented_as_native_or_partial_success():
    from scripts.replay_robomnist_lengths import require_scope

    r = dict(
        status="PASS_FIXED_DIRECTORY_LENGTH_SHORTCUT_DIAGNOSTIC_ONLY",
        unique_source_groups=2135,
        split_counts={"train": 1475, "validation": 660},
        metadata_estimator_fit=True,
        training_corpus_admission="NOT_ADMITTED",
        pretrained_models_loaded=False,
        teacher_training=False,
        gpu_used=False,
        new_payloads_opened=False,
        position_gt_opened=False,
        sealed_audit_opened=False,
        native_modality_quality_measured=False,
        elapsed_seconds=1.0,
        peak_cpu_bytes=100,
    )
    require_scope(r)
    with pytest.raises(ValueError):
        require_scope({**r, "native_modality_quality_measured": True})
    with pytest.raises(ValueError):
        require_scope({**r, "unique_source_groups": 2134})


def test_independent_scalar_metrics_keep_true_log_nll_and_full_class_brier():
    from scripts.replay_robomnist_lengths import scalar_metrics

    m = scalar_metrics(np.array([[0.0, -1000.0]]), np.array([1]))
    assert m["nll"] == 1000 and m["brier"] == 2 and m["ece"] == 1


def test_replay_metric_mismatch_cannot_be_hidden_by_accuracy_agreement():
    from scripts.replay_robomnist_lengths import compare_metrics

    a = {"count": 10, "accuracy": 0.5, "nll": 1.0}
    compare_metrics(a, a)
    with pytest.raises(ValueError):
        compare_metrics(a, {**a, "nll": 1.1})


def test_group_order_and_split_hashes_must_be_rederived_not_copied_from_report():
    from scripts.replay_robomnist_lengths import accounting, require_accounting

    rows = [
        dict(group_id="g1", split="train", repetition=1, class_target=0),
        dict(group_id="g2", split="validation", repetition=2, class_target=1),
    ]
    report = accounting(rows)
    require_accounting(report, rows)
    with pytest.raises(ValueError):
        require_accounting({**report, "group_order_sha256": "0" * 64}, rows)


def test_main_report_hash_binds_original_read_bytes_and_late_report_change_fails(tmp_path):
    import hashlib

    from scripts.replay_robomnist_lengths import read_report, recheck_bindings

    path = tmp_path / "result.json"
    blob = b'{ "status" : "fixed" }\n'
    path.write_bytes(blob)
    report, pin = read_report(path)
    assert report == {"status": "fixed"} and pin == hashlib.sha256(blob).hexdigest()
    recheck_bindings({str(path): pin})
    path.write_bytes(b'{"status":"changed"}')
    with pytest.raises(ValueError):
        recheck_bindings({str(path): pin})


def test_late_product_change_is_included_in_full_terminal_evidence_check(tmp_path):
    import hashlib

    from scripts.replay_robomnist_lengths import recheck_bindings

    path = tmp_path / "arrays.npz"
    path.write_bytes(b"original")
    pins = {str(path): hashlib.sha256(b"original").hexdigest()}
    path.write_bytes(b"changed after numerical verification")
    with pytest.raises(ValueError):
        recheck_bindings(pins)
