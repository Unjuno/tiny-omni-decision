import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tiny_omni_decision.decision import FrozenFeatureCandidateScorer  # noqa: E402
from tiny_omni_decision.native_feature_decision import (  # noqa: E402
    decide_single_modality_from_cache,
)
from tiny_omni_decision.observations import ObservationBatch  # noqa: E402


def make_batch(
    tmp_path, values, valid_mask, *, modality="image", cache_name="image-features.npz"
):
    path = tmp_path / cache_name
    timestamps = np.arange(len(values), dtype=np.float64) * 100.0
    np.savez(path, values=np.asarray(values, dtype=np.float32), timestamps_ms=timestamps,
             valid_mask=np.asarray(valid_mask, dtype=np.bool_))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ObservationBatch.model_validate({
        "schema_version": "1.0",
        "criterion": None,
        "observations": [{
            "modality": modality,
            "source_id": "camera-frame-cache",
            "payload_ref": str(path),
            "payload_sha256": digest,
            "clock_id": "capture-1",
            "timestamps_ms": timestamps.tolist(),
            "valid_mask": valid_mask,
            **({"channels": [{"name": "acc_x", "unit": "g", "scale": 1.0,
                              "offset": 0.0, "valid_range": [-16.0, 16.0]}]}
               if modality == "sensor" else {}),
        }],
        "options": [
            {"option_id": "turn-left", "attributes": {"left": 1, "right": 0}},
            {"option_id": "turn-right", "attributes": {"left": 0, "right": 1}},
        ],
    })


@pytest.mark.parametrize("modality", ["audio", "video", "image", "screen", "sensor"])
def test_cached_single_modality_decision_is_text_free_and_option_order_equivariant(
    tmp_path, modality
):
    torch.manual_seed(17)
    model = FrozenFeatureCandidateScorer(feature_size=4, hidden_size=8).eval()
    batch = make_batch(tmp_path, [[1., 2.], [1e6, -1e6], [3., 4.]], [True, False, True],
                       modality=modality)

    result = decide_single_modality_from_cache(
        model, batch, request_id="image-1", feature_width=2,
        option_attribute_keys=("left", "right"),
    )
    reordered = decide_single_modality_from_cache(
        model, batch.model_copy(update={"options": batch.options[::-1]}),
        request_id="image-1", feature_width=2,
        option_attribute_keys=("left", "right"),
    )

    assert result["status"] == "selected"
    assert result["modality"] == modality
    assert [row["id"] for row in result["options"]] == ["turn-left", "turn-right"]
    assert result["selected_option_id"] == reordered["selected_option_id"]
    assert {row["id"]: row["probability"] for row in result["options"]} == {
        row["id"]: row["probability"] for row in reordered["options"]
    }
    json.dumps(result, allow_nan=False)


def test_masked_cache_rows_do_not_change_single_modality_decision(tmp_path):
    torch.manual_seed(23)
    model = FrozenFeatureCandidateScorer(feature_size=4, hidden_size=8).eval()
    first = make_batch(tmp_path, [[1., 2.], [999., -999.], [3., 4.]], [True, False, True])
    second = make_batch(tmp_path, [[1., 2.], [-5e8, 7e8], [3., 4.]], [True, False, True],
                        cache_name="changed-masked-row.npz")
    kwargs = dict(request_id="same-observation", feature_width=2,
                  option_attribute_keys=("left", "right"))

    one = decide_single_modality_from_cache(model, first, **kwargs)
    two = decide_single_modality_from_cache(model, second, **kwargs)

    assert [row["probability"] for row in one["options"]] == [
        row["probability"] for row in two["options"]
    ]


def test_missing_features_abstain_and_invalid_text_or_attributes_fail_closed(tmp_path):
    torch.manual_seed(17)
    model = FrozenFeatureCandidateScorer(feature_size=4, hidden_size=8).eval()
    batch = make_batch(tmp_path, [[9., 9.], [8., 8.]], [False, False])
    kwargs = dict(request_id="missing-image", feature_width=2,
                  option_attribute_keys=("left", "right"))
    missing = decide_single_modality_from_cache(model, batch, **kwargs)
    assert missing["status"] == "abstain"
    assert missing["selected_option_id"] is None
    assert all(row["probability"] is None for row in missing["options"])

    text_data = batch.model_dump()
    text_data["criterion"] = {"kind": "text", "value": "choose left"}
    with_text = ObservationBatch.model_validate(text_data)
    with pytest.raises(ValueError, match="criterion"):
        decide_single_modality_from_cache(model, with_text, **kwargs)

    with pytest.raises(ValueError, match="temperature"):
        decide_single_modality_from_cache(model, batch, **(kwargs | {"temperature": "hot"}))

    data = batch.model_dump()
    data["options"][0]["attributes"]["left"] = "left"
    with pytest.raises(ValueError, match="numeric"):
        decide_single_modality_from_cache(model, ObservationBatch.model_validate(data), **kwargs)


def test_native_feature_decision_import_does_not_load_transformers():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'src'); "
            "import tiny_omni_decision.native_feature_decision; "
            "assert 'transformers' not in sys.modules",
        ],
        check=True,
    )
