import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _rows_and_features():
    kinds = ("temporal_descriptive", "explanatory", "predictive", "counterfactual")
    rows = []
    text = []
    targets = []
    videos = []
    for kind_index, kind in enumerate(kinds):
        for index in range(2):
            options = ["a", "b", "c"] if kind == "temporal_descriptive" else ["wrong", "correct"]
            rows.append(
                {
                    "id": f"{kind}-{index}",
                    "scene_index": kind_index * 2 + index,
                    "question_type": kind,
                    "question": "synthetic test question",
                    "options": options,
                    "target": options[index % len(options)],
                }
            )
            text.append(torch.randn(len(options), 12))
            targets.append(index % len(options))
            videos.append(torch.randn(8, 6))
    return rows, text, targets, torch.stack(videos)


def test_temporal_fit_saves_selected_checkpoint_and_variable_choice_predictions(tmp_path):
    from scripts.train_clevrer_temporal_mix import _fit_variant
    from tiny_omni_decision.temporal_candidate import TemporalCandidateScorer

    rows, text, targets, videos = _rows_and_features()
    config = {
        "training": {
            "learning_rate": 0.001,
            "weight_decay": 0.001,
            "batch_questions": 4,
            "epochs": 1,
        },
        "model": {
            "text_feature_size": 12,
            "video_feature_size": 6,
            "temporal_hidden_size": 4,
            "scorer_hidden_size": 5,
        },
    }
    model = TemporalCandidateScorer(
        text_feature_size=12,
        video_feature_size=6,
        temporal_hidden_size=4,
        scorer_hidden_size=5,
    )

    result = _fit_variant(
        name="videoprism_bilstm_text",
        model=model,
        train_text=text,
        train_targets=targets,
        train_rows=rows,
        validation_text=text,
        validation_targets=targets,
        validation_rows=rows,
        train_videos=videos,
        validation_videos=videos,
        config=config,
        seed=17,
        device=torch.device("cpu"),
        run_dir=tmp_path,
    )

    assert result["best_epoch"] == 1
    assert result["optimizer_updates"] == 2
    predictions = [
        json.loads(line)
        for line in (tmp_path / "videoprism_bilstm_text-validation-predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(predictions) == len(rows)
    assert [len(row["probabilities"]) for row in predictions[:3]] == [3, 3, 2]
    assert result["best_checkpoint_sha256"]
    assert result["final_checkpoint_sha256"]


def test_paired_scene_bootstrap_is_zero_for_identical_predictions():
    from scripts.train_clevrer_temporal_mix import _paired_scene_bootstrap

    rows, _, targets, _ = _rows_and_features()
    predictions = []
    for row, target in zip(rows, targets, strict=True):
        width = len(row["options"])
        values = [1.0 / width] * width
        predictions.append(
            {
                "id": row["id"],
                "scene_index": row["scene_index"],
                "question_type": row["question_type"],
                "target_index": target,
                "probabilities": values,
            }
        )

    result = _paired_scene_bootstrap(predictions, predictions, seed=17, replicates=20)

    assert result["macro_question_type"]["accuracy"] == [0.0, 0.0]
    assert result["by_question_type"]["predictive"]["nll"] == [0.0, 0.0]


def test_fit_requires_valid_hotugpu_handoff(monkeypatch):
    from scripts.train_clevrer_temporal_mix import GPU_GATE_ENV, _require_gpu_gate

    config = {
        "gpu_gate": {
            "required": True,
            "block_if_free_mib": 14000,
            "min_child_free_mib": 8000,
            "min_reserved_mib": 3200,
            "max_start_utilization_pct": 5,
        }
    }
    for key, value in {
        "reserved_mib": "3200",
        "free_mib": "13000",
        "total_mib": "16384",
        "utilization_pct": "0",
        "block_if_free_mib": "14000",
        "min_child_free_mib": "8000",
    }.items():
        monkeypatch.setenv(GPU_GATE_ENV[key], value)

    assert _require_gpu_gate(config)["reserved_mib"] == 3200

    monkeypatch.delenv(GPU_GATE_ENV["reserved_mib"])
    with pytest.raises(RuntimeError, match="HotUGPU gate"):
        _require_gpu_gate(config)
