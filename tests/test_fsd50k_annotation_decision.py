from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.build_fsd50k_annotation_decision_manifest import (  # noqa: E402
    DATASET_REVISION,
    TARGET_LABELS,
    canonical_sha256,
    summarize_split,
)
from scripts.train_fsd50k_annotation_decision import (  # noqa: E402
    EXPECTED_READOUT_PARAMETERS,
    FEATURE_WIDTH,
    expand_clip_batch,
    load_feature_data,
    macro_metrics,
    metric_summary,
    run_training,
    sha256_file,
    typed_binary_options,
)
from tiny_omni_decision.native_feature_decision import NativeFeatureDecisionScorer  # noqa: E402


def test_typed_binary_options_are_fixed_and_candidate_order_equivariant():
    options = typed_binary_options()
    assert options.shape == (4, 2, 6)
    assert options.dtype == torch.float32
    assert torch.equal(options.sum(-1), torch.full((4, 2), 2.0))

    model = NativeFeatureDecisionScorer(FEATURE_WIDTH, 6, 64).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_READOUT_PARAMETERS
    observation = torch.randn(4, FEATURE_WIDTH)
    valid = torch.ones((4, 2), dtype=torch.bool)
    with torch.inference_mode():
        logits = model(observation, options, valid)
        flipped = model(observation, options.flip(1), valid)
    torch.testing.assert_close(logits, flipped.flip(1))


def test_expand_clip_batch_encodes_four_binary_questions_without_text():
    options = typed_binary_options()
    features = torch.zeros((2, FEATURE_WIDTH))
    targets = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    obs, candidate_options, valid, labels = expand_clip_batch(features, targets, options)
    assert obs.shape == (8, FEATURE_WIDTH)
    assert candidate_options.shape == (8, 2, 6)
    assert valid.shape == (8, 2) and valid.all()
    assert labels.tolist() == [0, 1, 0, 1, 1, 0, 1, 0]
    assert torch.equal(candidate_options[0], options[0])
    assert torch.equal(candidate_options[1], options[1])


def test_binary_metrics_and_macro_are_finite_and_equal_weighted():
    logits = torch.tensor([[8.0, -8.0], [-8.0, 8.0]])
    targets = torch.tensor([0, 1])
    result = metric_summary(logits, targets)
    assert result["count"] == 2
    assert result["accuracy"] == 1.0
    assert result["nll"] < 1e-5
    assert result["brier"] < 1e-10
    assert 0.0 <= result["ece_15"] <= 1.0
    assert result["positive_precision"] == 1.0
    assert result["positive_recall"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["positive_support"] == 1

    values = {
        label: {"accuracy": float(index), "nll": 4.0 - index, "brier": 0.2, "ece_15": 0.1,
                "mean_confidence": 0.7, "positive_precision": 0.5,
                "positive_recall": 0.25, "balanced_accuracy": 0.375}
        for index, label in enumerate(TARGET_LABELS)
    }
    macro = macro_metrics(values)
    assert macro["accuracy"] == pytest.approx(1.5)
    assert macro["nll"] == pytest.approx(2.5)
    assert macro["positive_recall"] == pytest.approx(0.25)
    assert macro["balanced_accuracy"] == pytest.approx(0.375)


def test_positive_metrics_expose_majority_class_accuracy_hiding_missed_labels():
    logits = torch.tensor([[8.0, -8.0]] * 9)
    targets = torch.tensor([0] * 8 + [1])

    result = metric_summary(logits, targets)

    assert result["accuracy"] == pytest.approx(8 / 9)
    assert result["positive_recall"] == 0.0
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["positive_support"] == 1


def _write_feature_fixture(root):
    preflight_dir = root / "preflight"
    feature_dir = root / "features"
    preflight_dir.mkdir()
    feature_dir.mkdir()
    splits = {}
    for split in ("train", "calibration", "validation"):
        splits[split] = []
        offset = {"train": 0, "calibration": 2, "validation": 4}[split]
        for sample_index in range(offset, offset + 2):
            row = {
                "sample_id": f"{sample_index:06d}",
                "source_id": f"fsd:{sample_index}",
                "group_id": f"uploader-{sample_index}",
                "split": split,
                "license": "http://creativecommons.org/licenses/by/3.0/",
                "targets": {
                    label: bool((sample_index + label_index) % 2)
                    for label_index, label in enumerate(TARGET_LABELS)
                },
            }
            splits[split].append(row)
        (preflight_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in splits[split]),
            encoding="utf-8",
        )
    summaries = {split: summarize_split(rows) for split, rows in splits.items()}
    source_sha = {"manifest": "b" * 64}
    manifest_hash = canonical_sha256({"splits": summaries, "source_sha256": source_sha})
    (preflight_dir / "preflight.json").write_text(
        json.dumps(
            {
                "status": "PREFLIGHT_PASS",
                "dataset_revision": DATASET_REVISION,
                "manifest_sha256": manifest_hash,
                "source_sha256": source_sha,
                "splits": summaries,
                "official_eval_labels_read": False,
                "official_eval_media_acquired": False,
            }
        ),
        encoding="utf-8",
    )
    verification_path = root / "acquisition-verification.json"
    verification_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "preflight_manifest_sha256": manifest_hash,
                "official_eval_labels_read": False,
                "official_eval_media_acquired": False,
            }
        ),
        encoding="utf-8",
    )
    all_rows = [row for split in ("train", "calibration", "validation") for row in splits[split]]
    feature_rows = [
        {
            **row,
            "feature_index": index,
            "media_sha256": f"{index:064x}",
        }
        for index, row in enumerate(all_rows)
    ]
    features_path = feature_dir / "features.npy"
    np.save(features_path, np.zeros((len(feature_rows), FEATURE_WIDTH), dtype=np.float32))
    rows_path = feature_dir / "feature-rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in feature_rows),
        encoding="utf-8",
    )
    (feature_dir / "feature-run.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "preflight_manifest_sha256": manifest_hash,
                "acquisition_verification_sha256": sha256_file(verification_path),
                "sample_order_sha256": canonical_sha256([row["sample_id"] for row in all_rows]),
                "features_sha256": sha256_file(features_path),
                "feature_rows_sha256": sha256_file(rows_path),
                "encoder_revision": "a425fdce92572e602a1d5634799bd9f1f2efa806",
                "encoder_weights_sha256": (
                    "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e"
                ),
                "feature_width": FEATURE_WIDTH,
                "feature_dtype": "float32",
                "temporal_policy": "whole_clip_native_audio_resampled_to_32khz",
            }
        ),
        encoding="utf-8",
    )
    return preflight_dir, verification_path, feature_dir


def test_feature_loader_requires_pinned_complete_contract_and_hashes(tmp_path):
    preflight, verification, feature_dir = _write_feature_fixture(tmp_path)
    features, rows, _ = load_feature_data(preflight, verification, feature_dir)
    assert features.shape == (6, FEATURE_WIDTH)
    assert len(rows) == 6

    with (feature_dir / "features.npy").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="feature receipt"):
        load_feature_data(preflight, verification, feature_dir)


def test_cpu_training_saves_best_final_and_reloads_best(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rng = np.random.default_rng(17)
        labels_by_split = {"train": 16, "calibration": 8, "validation": 12}
        rows = []
        feature_rows = []
        for split, count in labels_by_split.items():
            for index in range(count):
                targets = {
                    label: bool((index + label_index) % 2)
                    for label_index, label in enumerate(TARGET_LABELS)
                }
                signal = np.asarray(
                    [1.0 if value else -1.0 for value in targets.values()],
                    dtype=np.float32,
                )
                feature = np.zeros(FEATURE_WIDTH, dtype=np.float32)
                feature[:4] = signal
                feature += rng.normal(0.0, 0.01, size=FEATURE_WIDTH).astype(np.float32)
                feature_rows.append(feature)
                rows.append(
                    {
                        "sample_id": str(len(rows)),
                        "source_id": f"fsd:{len(rows)}",
                        "group_id": f"uploader-{split}-{index}",
                        "split": split,
                        "targets": targets,
                    }
                )
        output = tmp_path / "run"
        result = run_training(
            np.stack(feature_rows),
            rows,
            {"preflight_manifest_sha256": "a" * 64},
            output,
            device="cpu",
        )
        assert result["status"] == "PASS"
        assert result["best_checkpoint_reload_verified"] is True
        assert result["readout_parameters"] == EXPECTED_READOUT_PARAMETERS
        assert result["official_eval_used"] is False
        assert (output / "best-checkpoint.pt").is_file()
        assert (output / "final-checkpoint.pt").is_file()
        history_path = output / "training-history.jsonl"
        history = [json.loads(line) for line in history_path.read_text().splitlines()]
        assert len(history) == 20
        assert sum(row["selected"] for row in history) >= 1
        assert sum(row["selected"] for row in history) == 1
        prediction_path = output / "selected-validation-predictions.jsonl"
        assert len(prediction_path.read_text().splitlines()) == 12 * 4
    finally:
        torch.set_num_threads(previous_threads)
