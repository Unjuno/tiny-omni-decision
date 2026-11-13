"""Integrity tests for the frozen-feature AVATAR Decision readout."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.fit_avatar_decision_readout import (
    _fit_candidate,
    _gated_model_class,
    _run,
    _score_audio_category_candidates,
    assemble_split,
    compute_grouped_metrics,
    compute_metrics,
    group_disjoint_donor_indices,
    load_audio_feature_cache,
    load_video_feature_cache,
)


def _example(example_id: str, split: str, audio_key: str, option_keys: list[str]):
    options = [
        {
            "option_id": f"o{index}",
            "video_asset_key": key,
            "video_timestamp_seconds": float(index + 1),
        }
        for index, key in enumerate(option_keys)
    ]
    return {
        "example_id": example_id,
        "split": split,
        "audio_asset_key": audio_key,
        "audio_window_center_seconds": 1.0,
        "audio_window_duration_seconds": 2.0,
        "audio_window_left_padding_seconds": 0.0,
        "audio_window_right_padding_seconds": 0.0,
        "options": options,
        "target_option_id": "o2",
    }


def _sidecar(row, group: str):
    option_groups = [f"option-{i}" for i in range(4)]
    option_groups[2] = group
    option_categories = [f"category-{i}" for i in range(4)]
    return {
        "example_id": row["example_id"],
        "target_option_index": 2,
        "option_asset_keys": [option["video_asset_key"] for option in row["options"]],
        "target_source_group": group,
        "option_source_groups": option_groups,
        "target_category": option_categories[2],
        "option_categories": option_categories,
    }


def test_assemble_split_tracks_all_source_groups_and_exact_feature_assets():
    row = _example("e1", "train", "a1", ["v1", "v2", "v3", "v4"])
    sidecar = _sidecar(row, "target-scene")
    result = assemble_split(
        [row],
        [sidecar],
        ["a1"],
        np.zeros((1, 3), dtype=np.float32),
        ["v1", "v2", "v3", "v4"],
        np.zeros((4, 2), dtype=np.float32),
        expected_split="train",
    )
    assert result["targets"].tolist() == [2]
    assert result["audio_asset_ids"] == ["a1"]
    assert result["video_option_asset_ids"] == [["v1", "v2", "v3", "v4"]]
    assert result["metadata"].shape == (1, 4, 5)
    assert result["option_categories"] == [["category-0", "category-1", "category-2", "category-3"]]
    assert result["source_groups"] == {"target-scene", "option-0", "option-1", "option-3"}
    assert result["audio_asset_keys"] == {"a1"}
    assert result["video_asset_keys"] == {"v1", "v2", "v3", "v4"}


def test_assemble_split_rejects_sidecar_option_order_mismatch():
    row = _example("e1", "train", "a1", ["v1", "v2", "v3", "v4"])
    sidecar = _sidecar(row, "target-scene")
    sidecar["option_asset_keys"] = list(reversed(sidecar["option_asset_keys"]))
    with pytest.raises(ValueError, match="option order"):
        assemble_split(
            [row],
            [sidecar],
            ["a1"],
            np.zeros((1, 3), dtype=np.float32),
            ["v1", "v2", "v3", "v4"],
            np.zeros((4, 2), dtype=np.float32),
            expected_split="train",
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("option_source_groups", ["same", "same", "target-scene", "other"], "source group"),
        ("option_categories", ["same", "same", "category-2", "category-3"], "distinct categories"),
    ],
)
def test_assemble_split_rejects_duplicate_option_group_or_category(field, replacement, message):
    row = _example("e1", "train", "a1", ["v1", "v2", "v3", "v4"])
    sidecar = _sidecar(row, "target-scene")
    sidecar[field] = replacement
    with pytest.raises(ValueError, match=message):
        assemble_split(
            [row],
            [sidecar],
            ["a1"],
            np.zeros((1, 3), dtype=np.float32),
            ["v1", "v2", "v3", "v4"],
            np.zeros((4, 2), dtype=np.float32),
            expected_split="train",
        )


def test_group_disjoint_donor_assignment_never_uses_same_group():
    groups = ["a", "a", "b", "b", "c", "d"]
    first = group_disjoint_donor_indices(groups, seed=17)
    second = group_disjoint_donor_indices(groups, seed=17)
    assert np.array_equal(first, second)
    assert all(groups[index] != groups[donor] for index, donor in enumerate(first))


def test_group_disjoint_donor_assignment_rejects_one_group():
    with pytest.raises(ValueError, match="at least two source groups"):
        group_disjoint_donor_indices(["same", "same"])


def test_grouped_metrics_reports_macro_over_categories():
    probabilities = np.asarray(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.7, 0.1, 0.1, 0.1],
            [0.7, 0.1, 0.1, 0.1],
            [0.7, 0.1, 0.1, 0.1],
        ]
    )
    metrics = compute_grouped_metrics(
        probabilities, np.asarray([0, 0, 1, 1]), ["easy", "easy", "hard", "hard"]
    )
    assert metrics["group_count"] == 2
    assert metrics["by_group"]["easy"]["accuracy"] == 1.0
    assert metrics["by_group"]["hard"]["accuracy"] == 0.0
    assert metrics["macro_over_groups"]["accuracy"] == 0.5


def test_audio_category_control_scores_unseen_option_classes_as_unavailable():
    logits = np.asarray([[2.0, -1.0]], dtype=np.float32)
    scores = _score_audio_category_candidates(
        logits, [["seen", "unseen", "other", "seen"]], {"seen": 0, "other": 1}
    )
    assert scores.shape == (1, 4)
    assert scores[0, 1] == -30.0
    assert scores[0, 0] == 2.0


def test_audio_cache_checks_centers_and_records_real_padding(tmp_path):
    path = tmp_path / "audio.npz"
    np.savez_compressed(
        path,
        asset_keys=np.asarray(["a", "b"]),
        centers_seconds=np.asarray([1.0, 2.0]),
        features=np.zeros((2, 384), dtype=np.float32),
        padding_samples=np.asarray([[0, 0], [0, 96]], dtype=np.int64),
        declared_padding_samples=np.asarray([[0, 0], [0, 64]], dtype=np.int64),
    )
    expected = {"a": (1.0, (0, 0)), "b": (2.0, (0, 64))}
    keys, features, diagnostics = load_audio_feature_cache(path, expected)
    assert keys == ["a", "b"]
    assert features.shape == (2, 384)
    assert diagnostics["actual_padding_mismatch_count"] == 1
    assert diagnostics["audio_shorter_than_annotation_count"] == 1
    expected["b"] = (2.1, (0, 64))
    with pytest.raises(ValueError, match="center differs"):
        load_audio_feature_cache(path, expected)


def test_video_cache_checks_frozen_timestamps_and_frame_alignment(tmp_path):
    path = tmp_path / "video.npz"
    np.savez_compressed(
        path,
        asset_keys=np.asarray(["v1", "v2"]),
        timestamps=np.asarray([1.25, 2.5]),
        features=np.zeros((2, 768), dtype=np.float32),
        max_frame_time_error_seconds=np.asarray([0.01, 0.06], dtype=np.float32),
    )
    keys, features, diagnostics = load_video_feature_cache(path, {"v1": 1.25, "v2": 2.5})
    assert keys == ["v1", "v2"]
    assert features.shape == (2, 768)
    assert diagnostics["timestamps_match_frozen_input"] is True
    assert diagnostics["frame_time_error_max_seconds"] == pytest.approx(0.06, abs=1e-6)
    with pytest.raises(ValueError, match="timestamp differs"):
        load_video_feature_cache(path, {"v1": 1.25, "v2": 2.6})


def test_uniform_metrics_have_expected_chance_accuracy_and_calibration():
    probabilities = np.full((8, 4), 0.25)
    metrics = compute_metrics(probabilities, np.asarray([0, 1, 2, 3] * 2))
    assert metrics["accuracy"] == 0.25
    assert metrics["nll"] == pytest.approx(np.log(4))
    assert metrics["brier"] == pytest.approx(0.75)
    assert metrics["ece_15"] == pytest.approx(0.0)


def test_cpu_fit_candidate_keeps_best_and_last_checkpoints_separate(tmp_path):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(17)
    train_x = rng.normal(size=(16, 4, 3)).astype(np.float32)
    train_y = np.arange(16, dtype=np.int64) % 4
    validation_x = rng.normal(size=(8, 4, 3)).astype(np.float32)
    validation_y = np.arange(8, dtype=np.int64) % 4
    best_path = tmp_path / "best-state.pt"
    result = _fit_candidate(
        torch,
        train_x,
        train_y,
        validation_x,
        validation_y,
        output_path=best_path,
        device=torch.device("cpu"),
    )
    assert best_path.is_file()
    assert (tmp_path / "last-state.pt").is_file()
    assert result["best_epoch"] <= result["epochs_run"]
    assert result["selected_validation_metrics"]["count"] == 8
    assert result["history"][result["best_epoch"] - 1]["fixed_train_evaluation"]
    saved_best = torch.load(best_path, map_location="cpu", weights_only=True)
    saved_last = torch.load(tmp_path / "last-state.pt", map_location="cpu", weights_only=True)
    assert saved_best.keys() == result["model"].state_dict().keys()
    assert saved_last.keys() == saved_best.keys()
    assert all(
        torch.equal(saved_best[key], result["model"].state_dict()[key].cpu()) for key in saved_best
    )


def test_cpu_end_to_end_readout_writes_all_control_arms(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(17)
    all_audio_keys, all_audio_features, all_audio_centers = [], [], []
    all_video_keys, all_video_features, all_video_timestamps = [], [], []
    split_rows = {}
    split_sidecars = {}
    for split, example_count in (("train", 16), ("validation", 8)):
        rows, sidecars = [], []
        for example_index in range(example_count):
            example_id = f"{split}-{example_index}"
            audio_key = f"audio-{split}-{example_index}"
            all_audio_keys.append(audio_key)
            all_audio_features.append(rng.normal(size=384).astype(np.float32))
            all_audio_centers.append(1.0)
            target = example_index % 4
            option_categories = [f"category-{i}" for i in range(4)]
            options, option_groups = [], []
            for option_index in range(4):
                video_key = f"video-{split}-{example_index}-{option_index}"
                all_video_keys.append(video_key)
                all_video_features.append(rng.normal(size=768).astype(np.float32))
                all_video_timestamps.append(float(option_index + 1))
                options.append(
                    {
                        "option_id": f"o{option_index}",
                        "video_asset_key": video_key,
                        "video_timestamp_seconds": float(option_index + 1),
                    }
                )
                option_groups.append(f"option-group-{split}-{example_index}-{option_index}")
            target_group = f"target-group-{split}-{example_index}"
            audio_group = f"audio-group-{split}-{example_index}"
            option_groups[target] = target_group
            rows.append(
                {
                    "example_id": example_id,
                    "split": split,
                    "audio_asset_key": audio_key,
                    "audio_window_center_seconds": 1.0,
                    "audio_window_duration_seconds": 2.0,
                    "audio_window_left_padding_seconds": 0.0,
                    "audio_window_right_padding_seconds": 0.0,
                    "options": options,
                    "target_option_id": f"o{target}",
                }
            )
            sidecars.append(
                {
                    "example_id": example_id,
                    "audio_source_group": audio_group,
                    "target_option_index": target,
                    "target_source_group": target_group,
                    "target_category": option_categories[target],
                    "option_source_groups": option_groups,
                    "option_categories": option_categories,
                    "option_asset_keys": [option["video_asset_key"] for option in options],
                }
            )
        split_rows[split] = rows
        split_sidecars[split] = sidecars

    def write_jsonl(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    paths = {}
    for split in ("train", "validation"):
        paths[f"{split}_input"] = tmp_path / f"{split}.jsonl"
        paths[f"{split}_sidecar"] = tmp_path / f"{split}-sidecar.jsonl"
        write_jsonl(paths[f"{split}_input"], split_rows[split])
        write_jsonl(paths[f"{split}_sidecar"], split_sidecars[split])
    audio_features_path = tmp_path / "audio-features.npz"
    video_features_path = tmp_path / "video-features.npz"
    np.savez_compressed(
        audio_features_path,
        asset_keys=np.asarray(all_audio_keys),
        centers_seconds=np.asarray(all_audio_centers),
        features=np.stack(all_audio_features),
        padding_samples=np.zeros((len(all_audio_keys), 2), dtype=np.int64),
        declared_padding_samples=np.zeros((len(all_audio_keys), 2), dtype=np.int64),
    )
    np.savez_compressed(
        video_features_path,
        asset_keys=np.asarray(all_video_keys),
        timestamps=np.asarray(all_video_timestamps),
        features=np.stack(all_video_features),
        max_frame_time_error_seconds=np.zeros(len(all_video_keys), dtype=np.float32),
    )
    output_dir = tmp_path / "readout-output"
    result = _run(
        SimpleNamespace(
            train_input=paths["train_input"],
            validation_input=paths["validation_input"],
            train_sidecar=paths["train_sidecar"],
            validation_sidecar=paths["validation_sidecar"],
            audio_features=audio_features_path,
            video_features=video_features_path,
            output_dir=output_dir,
            device="cpu",
        )
    )
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert result["status"] == report["status"] == "PASS"
    assert report["device"] == "cpu"
    assert report["sealed_audit_loaded"] is False
    assert report["validation_category_metrics"]["joint"]["group_count"] == 4
    assert report["validation_source_group_metrics"]["joint"]["group_count"] == 8
    predictions = json.loads(
        (output_dir / "validation-predictions.json").read_text(encoding="utf-8")
    )
    assert predictions[0]["audio_source_group"].startswith("audio-group-validation-")
    assert predictions[0]["target_source_group"].startswith("target-group-validation-")
    assert report["group_paired_bootstrap"]["joint_minus_video_only_accuracy"][
        "resampling_unit"
    ].startswith("audio query source-video group")
    assert {
        "option_position_prior",
        "metadata_only",
        "audio_only_oracle_category_match",
        "video_only",
        "joint_audio_video",
        "joint_audio_video_conditional_gate",
        "joint_gate_missing_audio",
        "joint_gate_missing_video",
    }.issubset(report["candidates"])
    gate_info = report["training"]["joint_conditional_gate"]
    assert gate_info["parameter_count"] == 82_177
    assert set(gate_info["gate_diagnostics"]) == {
        "audio_weight_mean",
        "audio_weight_p10",
        "audio_weight_p90",
        "fraction_saturated_below_0_05",
        "fraction_saturated_above_0_95",
    }
    assert predictions[0]["joint_conditional_gate"]["probabilities"]


def test_conditional_gate_is_differentiable_and_option_equivariant():
    torch = pytest.importorskip("torch")
    model = _gated_model_class(torch)()
    assert sum(parameter.numel() for parameter in model.parameters()) == 82_177
    values = torch.randn(3, 4, 1152)
    logits = model(values)
    gates = model.gates(values)
    assert logits.shape == (3, 4)
    assert gates.shape == (3, 4, 64)
    assert torch.isfinite(gates).all()
    order = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(model(values[:, order]), logits[:, order])
    logits.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
