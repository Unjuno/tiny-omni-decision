from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest


def _write_rows(archive, path, values):
    archive.writestr(path, "\n".join(" ".join(str(v) for v in row) for row in values))


def _synthetic_uci_archive():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        root = "UCI HAR Dataset"
        split = "train"
        labels = [1, 2, 3, 4, 5, 6]
        subjects = [1, 1, 2, 2, 3, 3]
        _write_rows(archive, f"{root}/{split}/y_{split}.txt", [[v] for v in labels])
        _write_rows(archive, f"{root}/{split}/subject_{split}.txt", [[v] for v in subjects])
        for channel_index, channel in enumerate(
            ("body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z")
        ):
            rows = [
                [subject + channel_index / 10 + point / 1000 for point in range(128)]
                for subject in subjects
            ]
            _write_rows(
                archive,
                f"{root}/{split}/Inertial Signals/{channel}_{split}.txt",
                rows,
            )
        # This invalid test payload must not be read by the train-split loader.
        archive.writestr(f"{root}/test/y_test.txt", "not numeric")
    buffer.seek(0)
    return zipfile.ZipFile(buffer)


def test_loader_reads_fixed_six_channel_train_windows_without_touching_test_split():
    from scripts.uci_har_sensor_experiment import load_uci_har_split

    with _synthetic_uci_archive() as archive:
        values, labels, subjects = load_uci_har_split(archive, "train")

    assert values.shape == (6, 128, 6)
    assert labels.tolist() == [1, 2, 3, 4, 5, 6]
    assert subjects.tolist() == [1, 1, 2, 2, 3, 3]
    assert np.isfinite(values).all()
    assert values[0, 0].tolist() == pytest.approx([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])


def test_official_wrapper_opens_nested_archive_for_train_only(tmp_path):
    from scripts.uci_har_sensor_experiment import _open_uci_har_data, load_uci_har_split

    with _synthetic_uci_archive() as inner:
        nested_bytes = inner.fp.getvalue()
    outer_path = tmp_path / "uci-wrapper.zip"
    with zipfile.ZipFile(outer_path, "w") as outer:
        outer.writestr("UCI HAR Dataset.zip", nested_bytes)

    with _open_uci_har_data(outer_path) as data_archive:
        values, labels, subjects = load_uci_har_split(data_archive, "train")

    assert values.shape == (6, 128, 6)
    assert labels.tolist() == [1, 2, 3, 4, 5, 6]
    assert subjects.tolist() == [1, 1, 2, 2, 3, 3]


def test_train_validation_split_is_deterministic_and_subject_disjoint():
    from scripts.uci_har_sensor_experiment import split_train_subjects

    subjects = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
    train, validation = split_train_subjects(subjects, seed=17, validation_subject_count=4)
    repeated = split_train_subjects(subjects, seed=17, validation_subject_count=4)

    assert (train, validation) == repeated
    assert len(train) == 17 and len(validation) == 4
    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == sorted(subjects)


def test_group_split_rejects_invalid_subject_inputs():
    from scripts.uci_har_sensor_experiment import split_train_subjects

    with pytest.raises(ValueError, match="unique"):
        split_train_subjects([1, 1, 2, 3, 4], seed=17, validation_subject_count=2)
    with pytest.raises(ValueError, match="subject groups"):
        split_train_subjects([1, 2], seed=17, validation_subject_count=2)


def test_fit_and_test_subject_overlap_fails_closed():
    from scripts.uci_har_sensor_experiment import require_disjoint_subjects

    assert require_disjoint_subjects([1, 2, 3], [4, 5]) == 0
    with pytest.raises(ValueError, match="subject overlap"):
        require_disjoint_subjects([1, 2, 3], [3, 4])


def test_effective_config_records_explicit_epoch_budget():
    from scripts.uci_har_sensor_experiment import _training_config

    config = _training_config(
        "lstm",
        "a" * 64,
        [1, 2],
        [3],
        epochs=100,
        source_commit="b" * 40,
        script_sha256="c" * 64,
    )

    assert config["epochs"] == 100
    assert config["seed"] == 17
    assert config["compute_device"] == "cpu"


def test_fit_rejects_nonpositive_budget_before_creating_output(tmp_path):
    from scripts.uci_har_sensor_experiment import fit_phase

    output = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError, match="epochs must be positive"):
        fit_phase(tmp_path / "missing.zip", output, epochs=0)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["static", "lstm"])
def test_candidate_scorers_have_finite_forward_and_gradients(kind):
    import torch

    from scripts.uci_har_sensor_experiment import ACTIVITIES, _logits, _new_model
    from tiny_omni_decision.native_feature_decision import EnumOptionField, TypedOptionSchema
    from tiny_omni_decision.observations import NativeOption

    torch.manual_seed(17)
    schema = TypedOptionSchema(enums=(EnumOptionField("activity", ACTIVITIES),))
    options = [NativeOption(option_id=name, attributes={"activity": name}) for name in ACTIVITIES]
    option_features = schema.encode(options).float()
    values = torch.randn(2, 128, 6)
    model = _new_model(kind)
    logits = _logits(model, kind, values, option_features)
    assert logits.shape == (2, len(ACTIVITIES))
    assert torch.isfinite(logits).all()

    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()

    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_public_test_audit_requires_freeze_before_reading_test(tmp_path):
    from scripts.uci_har_sensor_experiment import audit_phase

    archive_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("UCI HAR Dataset/test/y_test.txt", "not numeric")

    output = tmp_path / "attempt"
    output.mkdir()
    with pytest.raises(FileNotFoundError, match="frozen candidate"):
        audit_phase(archive_path, output, receipt_dir=tmp_path / "receipts")
    assert not (output / "official-test-audit-started.json").exists()
    assert not (tmp_path / "receipts").exists()


def test_public_test_source_hash_cannot_be_reopened_in_another_output(tmp_path, monkeypatch):
    import scripts.uci_har_sensor_experiment as experiment

    archive_path = tmp_path / "same-source.zip"
    archive_path.write_bytes(b"archive bytes are not opened by this regression")
    output = tmp_path / "new-output"
    output.mkdir()
    monkeypatch.setattr(
        experiment,
        "_sha256",
        lambda path: (
            "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
            if path == archive_path
            else "f" * 64
        ),
    )

    with pytest.raises(FileExistsError, match="permanently closed"):
        experiment.audit_phase(archive_path, output, receipt_dir=tmp_path / "receipts")

    assert not (output / "official-test-audit-started.json").exists()
    assert not (tmp_path / "receipts").exists()


def test_public_test_audit_receipt_is_global_to_source_hash(tmp_path):
    from scripts.uci_har_sensor_experiment import (
        _claim_audit_source,
        _mark_audit_source_loaded,
    )

    freeze_path = tmp_path / "candidate-freeze.json"
    freeze_path.write_text('{"frozen_before_test_read": true}\n', encoding="utf-8")
    output = tmp_path / "attempt-one"
    output.mkdir()
    receipt_dir = tmp_path / "receipts"

    receipt = _claim_audit_source(
        "d" * 64,
        output=output,
        freeze_path=freeze_path,
        receipt_dir=receipt_dir,
    )
    assert receipt.is_file()
    _mark_audit_source_loaded(receipt)
    assert json.loads(receipt.read_text(encoding="utf-8"))["state"].startswith(
        "test_split_loaded"
    )
    with pytest.raises(FileExistsError, match="already claimed"):
        _claim_audit_source(
            "d" * 64,
            output=tmp_path / "attempt-two",
            freeze_path=freeze_path,
            receipt_dir=receipt_dir,
        )
