import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tiny_omni_decision.aligned_features import gather_aligned_features  # noqa: E402
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer  # noqa: E402
from tiny_omni_decision.evidence_decision import evidence_gated_json  # noqa: E402
from tiny_omni_decision.fusion_decision import FusionDecisionScorer  # noqa: E402
from tiny_omni_decision.native_decision import (  # noqa: E402
    numeric_option_features,
    run_text_free_single_modality_decision,
)
from tiny_omni_decision.observations import NativeOption, ObservationBatch  # noqa: E402


def _cache(tmp_path, filename, values, times=(100.0,), valid=(True,)):
    path = tmp_path / filename
    np.savez(
        path,
        values=np.asarray(values, dtype=np.float32),
        timestamps_ms=np.asarray(times, dtype=np.float64),
        valid_mask=np.asarray(valid, dtype=np.bool_),
    )
    return path


def _observation(modality, source_id, path, *, times=(100.0,), valid=(True,)):
    result = {
        "modality": modality,
        "source_id": source_id,
        "payload_ref": str(path),
        "payload_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "clock_id": "fixture-clock",
        "timestamps_ms": times,
        "valid_mask": valid,
    }
    if modality == "sensor":
        result["channels"] = [{"name": "force", "unit": "N", "scale": 1.0, "offset": 0.0}]
    return result


def _options(ids=("opt-k3", "opt-q8")):
    return [
        {"option_id": ids[0], "attributes": {"action_code": 0}},
        {"option_id": ids[1], "attributes": {"action_code": 1}},
    ]


def _gather_single(batch, modality):
    source_id = batch.observations[0].source_id
    width = 2
    return gather_aligned_features(
        batch,
        {modality: source_id},
        {modality: width},
        [100.0],
        max_age_ms=1.0,
    )


def _best_feature_lookup_accuracy(values, labels):
    groups = {}
    for value, label in zip(map(tuple, values), labels, strict=True):
        groups.setdefault(value, []).append(label)
    correct = sum(max(group.count(0), group.count(1)) for group in groups.values())
    return correct / len(labels)


@pytest.mark.parametrize("modality", ["audio", "video", "image", "screen", "sensor"])
def test_text_free_native_m1_smoke_emits_typed_probabilities(tmp_path, modality):
    path = _cache(tmp_path, "observation-17.npz", [[0.25, -0.5]])
    batch = ObservationBatch.model_validate(
        {
            "schema_version": "1.0",
            "observations": [_observation(modality, "source-17", path)],
            "options": _options(),
        }
    )
    assert batch.criterion is None
    assert batch.present_modalities == (modality,)
    gathered = _gather_single(batch, modality)
    torch.manual_seed(17)
    scorer = FrozenFeatureCandidateScorer(3, hidden_size=8).eval()

    record = json.loads(
        run_text_free_single_modality_decision(
            scorer,
            batch,
            gathered,
            modality=modality,
            option_field_names=["action_code"],
            option_texts=["Option X", "Option Y"],
            request_id="request-opaque-01",
        )
    )

    assert record["decision_type"] == "choice"
    assert record["selected_option_id"] in {"opt-k3", "opt-q8"}
    assert [option["id"] for option in record["options"]] == ["opt-k3", "opt-q8"]
    assert sum(option["probability"] for option in record["options"]) == pytest.approx(1.0)


def test_m1_metadata_ids_and_feature_filename_do_not_change_probabilities(tmp_path):
    values = [[0.25, -0.5]]
    first_path = _cache(tmp_path, "observation-17.npz", values)
    second_path = _cache(tmp_path, "opaque-source-004.npz", values)
    first_batch = ObservationBatch.model_validate(
        {
            "schema_version": "1.0",
            "observations": [_observation("video", "source-17", first_path)],
            "options": _options(),
        }
    )
    relabeled_batch = ObservationBatch.model_validate(
        {
            "schema_version": "1.0",
            "observations": [_observation("video", "capture-a", second_path)],
            "options": _options(("q8-RANDOM", "p3-RANDOM")),
        }
    )
    torch.manual_seed(17)
    scorer = FrozenFeatureCandidateScorer(3, hidden_size=8).eval()

    def infer(batch, option_texts, request_id):
        result = run_text_free_single_modality_decision(
            scorer,
            batch,
            _gather_single(batch, "video"),
            modality="video",
            option_field_names=["action_code"],
            option_texts=option_texts,
            request_id=request_id,
        )
        return json.loads(result)

    first = infer(first_batch, ["First", "Second"], "request-same")
    relabeled = infer(
        relabeled_batch,
        ["Unrelated wording", "Different display text"],
        "request-other",
    )
    assert [x["probability"] for x in first["options"]] == pytest.approx(
        [x["probability"] for x in relabeled["options"]]
    )
    assert first["request_id"] != relabeled["request_id"]
    first_index = next(
        index
        for index, option in enumerate(first["options"])
        if option["id"] == first["selected_option_id"]
    )
    relabeled_index = next(
        index
        for index, option in enumerate(relabeled["options"])
        if option["id"] == relabeled["selected_option_id"]
    )
    assert first_index == relabeled_index


def test_native_decision_runner_imports_no_text_encoder():
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    inherited = os.environ.get("PYTHONPATH", "")
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, (source_root, inherited))),
    }
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tiny_omni_decision.native_decision; "
            "assert not any(name == 'transformers' or name.startswith('transformers.') "
            "for name in sys.modules)",
        ],
        check=True,
        env=environment,
    )


def test_m1_requires_numeric_typed_options_and_abstains_without_feature(tmp_path):
    path = _cache(tmp_path, "sensor-cache.npz", [[np.nan, np.nan]], valid=(False,))
    batch = ObservationBatch.model_validate(
        {
            "schema_version": "1.0",
            "observations": [_observation("sensor", "sensor-stream", path, valid=(False,))],
            "options": _options(),
        }
    )
    scorer = FrozenFeatureCandidateScorer(3, hidden_size=8).eval()
    abstained = json.loads(
        run_text_free_single_modality_decision(
            scorer,
            batch,
            _gather_single(batch, "sensor"),
            modality="sensor",
            option_field_names=["action_code"],
            option_texts=["Action A", "Action B"],
            request_id="request-missing",
        )
    )
    assert abstained["status"] == "abstain"
    assert abstained["selected_option_id"] is None
    with pytest.raises(ValueError, match="numeric"):
        numeric_option_features(
            [NativeOption(option_id="x", attributes={"action_code": "left"})],
            ["action_code"],
        )


def test_synchronized_video_sensor_fixture_uses_both_masks_and_abstains_if_one_missing(
    tmp_path,
):
    times = (10.0, 20.0, 30.0, 40.0)
    video_values = [[0.0, 0.2], [0.0, 0.2], [1.0, 0.2], [1.0, 0.2]]
    sensor_values = [[0.0, 0.4], [1.0, 0.4], [0.0, 0.4], [1.0, 0.4]]
    targets = [
        int((video[0] > 0.5) != (sensor[0] > 0.5))
        for video, sensor in zip(video_values, sensor_values, strict=True)
    ]
    assert targets == [0, 1, 1, 0]
    assert sum(targets) == 2  # Balanced synthetic labels and all factor combinations.
    assert targets[0] != targets[1] and targets[2] != targets[3]
    assert targets[0] != targets[2] and targets[1] != targets[3]
    assert _best_feature_lookup_accuracy(video_values, targets) == 0.5
    assert _best_feature_lookup_accuracy(sensor_values, targets) == 0.5

    video_path = _cache(
        tmp_path, "video-features.npz", video_values, times=times, valid=(True,) * 4
    )
    sensor_path = _cache(
        tmp_path, "sensor-features.npz", sensor_values, times=times, valid=(True,) * 4
    )
    batch = ObservationBatch.model_validate(
        {
            "schema_version": "1.0",
            "observations": [
                _observation("video", "camera-opaque", video_path, times=times, valid=(True,) * 4),
                _observation(
                    "sensor", "sensor-opaque", sensor_path, times=times, valid=(True,) * 4
                ),
            ],
            "options": _options(),
        }
    )
    assert batch.criterion is None
    batch.require_shared_clock()
    gathered = gather_aligned_features(
        batch,
        {"video": "camera-opaque", "sensor": "sensor-opaque"},
        {"video": 2, "sensor": 2},
        times,
        max_age_ms=1.0,
    )
    torch.manual_seed(17)
    model = FusionDecisionScorer(
        {"video": 2, "sensor": 2}, option_dim=1, criterion_dim=1, width=4
    ).eval()
    option_values = torch.from_numpy(numeric_option_features(batch.options, ["action_code"]))
    model_calls = []

    def predict(row):
        model_calls.append(True)
        with torch.inference_mode():
            logits = model(
                {
                    name: torch.from_numpy(gathered["features"][name][row : row + 1])
                    for name in ("video", "sensor")
                },
                {
                    name: torch.from_numpy(gathered["available"][name][row : row + 1])
                    for name in ("video", "sensor")
                },
                option_values[None],
                torch.ones(1, 2, dtype=torch.bool),
            )
            return logits.softmax(-1)[0].tolist()

    options = [{"option_id": item.option_id, "class_id": item.option_id} for item in batch.options]
    outputs = []
    for row in range(len(times)):
        outputs.append(
            evidence_gated_json(
                lambda row=row: predict(row),
                [item.option_id for item in batch.options],
                options,
                required=["video", "sensor"],
                available={
                    name: bool(gathered["available"][name][row]) for name in ("video", "sensor")
                },
            )
        )
    assert [output["status"] for output in outputs] == ["selected"] * 4
    assert model_calls == [True] * 4
    assert all(
        sum(item["probability"] for item in output["options"]) == pytest.approx(1.0)
        for output in outputs
    )

    missing_sensor = evidence_gated_json(
        lambda: predict(3),
        [item.option_id for item in batch.options],
        options,
        required=["video", "sensor"],
        available={"video": True, "sensor": False},
    )
    assert missing_sensor["status"] == "abstain"
    assert model_calls == [True] * 4
