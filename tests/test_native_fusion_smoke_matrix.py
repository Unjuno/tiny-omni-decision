"""CPU contract smokes for text-free native inputs; not quality evaluations."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tiny_omni_decision.fusion_decision import FusionDecisionScorer  # noqa: E402
from tiny_omni_decision.native_feature_decision import (  # noqa: E402
    NativeFeatureDecisionScorer,
    NumericOptionField,
    TypedOptionSchema,
    decide_cached_native_feature,
    decide_cached_native_fusion,
)
from tiny_omni_decision.observations import ObservationBatch  # noqa: E402

MODALITIES = ("audio", "image", "screen", "video", "sensor")
OPTION_SCHEMA = TypedOptionSchema(numeric=(NumericOptionField("action_code"),))


def _write_cache(path, values, stamps=(10.0, 20.0), valid=(True, True)):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        values=np.asarray(values, dtype=np.float32),
        timestamps_ms=np.asarray(stamps, dtype=np.float64),
        valid_mask=np.asarray(valid, dtype=np.bool_),
    )
    return {
        "payload_ref": str(path),
        "payload_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "timestamps_ms": tuple(stamps),
        "valid_mask": tuple(valid),
    }


def _options():
    # Opaque IDs and numeric attributes keep labels/names out of the input.
    return [
        {"option_id": "opt-17", "attributes": {"action_code": 0}},
        {"option_id": "opt-42", "attributes": {"action_code": 1}},
    ]


def _batch(observations):
    return ObservationBatch.model_validate(
        {"schema_version": "1.0", "observations": observations, "options": _options()}
    )


@pytest.mark.parametrize("modality", MODALITIES)
def test_text_free_m1_cached_runner_smokes_each_native_modality(tmp_path, modality):
    torch.manual_seed(17)
    path = tmp_path / "payload-01.npz"
    cache = _write_cache(path, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    observation = {
        "modality": modality,
        "source_id": f"source-{modality}",
        "clock_id": "clock-1",
        **cache,
    }
    if modality == "sensor":
        observation["channels"] = [
            {"name": "level", "unit": "unitless", "scale": 1.0, "offset": 0.0}
        ]
    batch = _batch([observation])
    model = NativeFeatureDecisionScorer(feature_width=3, option_width=1, hidden_width=8).eval()

    result = decide_cached_native_feature(
        model,
        batch,
        modality=modality,
        query_time_ms=20.0,
        max_age_ms=5.0,
        option_schema=OPTION_SCHEMA,
    )

    assert result["status"] == "selected"
    assert result["selected_option_id"] in {"opt-17", "opt-42"}
    assert [row["option_id"] for row in result["options"]] == ["opt-17", "opt-42"]
    assert sum(row["probability"] for row in result["options"]) == pytest.approx(1.0)
    assert batch.criterion is None
    assert all(set(option.attributes) == {"action_code"} for option in batch.options)


def test_native_m1_import_does_not_load_a_text_encoder():
    code = """
import sys
sys.modules['transformers'] = None
sys.modules['sentence_transformers'] = None
sys.modules['tiny_omni_decision.text_runtime'] = None
from tiny_omni_decision.native_feature_decision import NativeFeatureDecisionScorer
assert NativeFeatureDecisionScorer is not None
"""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (source_root, env.get("PYTHONPATH", "")) if path
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr


def _av_batch(tmp_path, *, audio_values, audio_stamps=(10.0, 20.0), audio_valid=(True, True)):
    audio_cache = _write_cache(
        tmp_path / "stream-a.npz", audio_values, stamps=audio_stamps, valid=audio_valid
    )
    video_cache = _write_cache(
        tmp_path / "stream-b.npz", [[0.0, 1.0], [1.0, 0.0]], stamps=(10.0, 20.0)
    )
    common = {"clock_id": "shared-clock"}
    return _batch(
        [
            {"modality": "audio", "source_id": "mic-opaque", **common, **audio_cache},
            {"modality": "video", "source_id": "cam-opaque", **common, **video_cache},
        ]
    )


def _run_av(model, batch, *, max_age_ms=15.0):
    return decide_cached_native_fusion(
        model,
        batch,
        source_by_modality={"audio": "mic-opaque", "video": "cam-opaque"},
        query_time_ms=20.0,
        max_age_ms=max_age_ms,
        required_modalities=("audio", "video"),
        option_schema=OPTION_SCHEMA,
    )


def test_synchronized_m2_smoke_routes_matched_missing_shuffle_offset_and_conflict(tmp_path):
    torch.manual_seed(23)
    model = FusionDecisionScorer({"audio": 2, "video": 2}, 1, 1, width=8).eval()
    seen = []
    original_forward = model.forward

    def capture(features, available, options, option_valid, criterion=None, criterion_present=None):
        seen.append(
            {
                "audio": features["audio"].detach().cpu().clone(),
                "video": features["video"].detach().cpu().clone(),
                "available": {
                    name: value.detach().cpu().clone() for name, value in available.items()
                },
            }
        )
        return original_forward(
            features, available, options, option_valid, criterion, criterion_present
        )

    model.forward = capture

    matched = _run_av(
        model,
        _av_batch(tmp_path / "matched", audio_values=[[1.0, 0.0], [0.0, 1.0]]),
    )
    assert matched["status"] == "selected"
    assert seen[-1]["audio"].tolist() == [[0.0, 1.0]]
    assert seen[-1]["video"].tolist() == [[1.0, 0.0]]
    assert all(bool(mask.item()) for mask in seen[-1]["available"].values())

    calls_before_missing = len(seen)
    missing = _run_av(
        model,
        _av_batch(
            tmp_path / "missing",
            audio_values=[[1.0, 0.0], [0.0, 1.0]],
            audio_valid=(False, False),
        ),
    )
    assert missing["status"] == "abstain"
    assert missing["missing_modalities"] == ["audio"]
    assert len(seen) == calls_before_missing  # abstention precedes model inference

    shuffled = _run_av(
        model,
        _av_batch(tmp_path / "shuffled", audio_values=[[0.0, 1.0], [1.0, 0.0]]),
    )
    assert shuffled["status"] == "selected"
    assert seen[-1]["audio"].tolist() == [[1.0, 0.0]]
    assert seen[-1]["video"].tolist() == [[1.0, 0.0]]

    offset = _run_av(
        model,
        _av_batch(
            tmp_path / "offset",
            audio_values=[[1.0, 0.0], [0.0, 1.0]],
            audio_stamps=(0.0, 10.0),
        ),
        max_age_ms=15.0,
    )
    assert offset["status"] == "selected"
    assert seen[-1]["audio"].tolist() == [[0.0, 1.0]]
    assert seen[-1]["video"].tolist() == [[1.0, 0.0]]

    contradictory = _run_av(
        model,
        _av_batch(tmp_path / "contradictory", audio_values=[[0.0, 1.0], [1.0, 0.0]]),
    )
    assert contradictory["status"] == "selected"
    assert seen[-1]["audio"].tolist() == [[1.0, 0.0]]
    assert seen[-1]["video"].tolist() == [[1.0, 0.0]]

    stale = _run_av(
        model,
        _av_batch(
            tmp_path / "stale",
            audio_values=[[1.0, 0.0], [0.0, 1.0]],
            audio_stamps=(0.0, 10.0),
        ),
        max_age_ms=5.0,
    )
    assert stale["status"] == "abstain"
    assert stale["missing_modalities"] == ["audio"]

    # These fixtures verify cache routing and abstention only, not that a learned
    # model correctly resolves the deliberately conflicting signals.


def test_m2_option_permutation_preserves_id_probability_mapping(tmp_path):
    torch.manual_seed(29)
    model = FusionDecisionScorer({"audio": 2, "video": 2}, 1, 1, width=8).eval()
    batch = _av_batch(tmp_path, audio_values=[[0.0, 1.0], [1.0, 0.0]])
    first = _run_av(model, batch)
    reordered = batch.model_copy(update={"options": tuple(reversed(batch.options))})
    second = _run_av(model, reordered)

    p_first = {row["option_id"]: row["probability"] for row in first["options"]}
    p_second = {row["option_id"]: row["probability"] for row in second["options"]}
    assert p_first.keys() == p_second.keys()
    for option_id in p_first:
        assert p_first[option_id] == pytest.approx(p_second[option_id], abs=1e-7)
