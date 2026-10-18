import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("PIL")
pytest.importorskip("numpy")


def _fixture_rows():
    from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES

    labels = {"id": "image-1", "source_split": "official-public-val"}
    rows = []
    for task, options in CLEVR4_TAXONOMIES.items():
        target = options[0]
        labels[task] = target
        probabilities = [1.0] + [0.0] * (len(options) - 1)
        rows.append(
            {
                "id": f"image-1:{task}",
                "image_id": "image-1",
                "taxonomy": task,
                "options": options,
                "target": target,
                "global_probabilities": probabilities,
                "spatial_probabilities": probabilities,
            }
        )
    return [labels], rows


def test_saved_predictions_are_reaggregated_and_paired_by_image():
    from scripts.finalize_dinov2_spatial_public_val import summarize_predictions

    labels, rows = _fixture_rows()
    metrics, differences = summarize_predictions(labels, rows, expected_images=1)
    assert metrics["global"]["accuracy"] == 1.0
    assert metrics["spatial"]["accuracy"] == 1.0
    assert metrics["spatial"]["ece"] == 0.0
    assert all(values.tolist() == [0.0] for values in differences.values())


def test_saved_prediction_target_mismatch_fails_closed():
    from scripts.finalize_dinov2_spatial_public_val import summarize_predictions

    labels, rows = _fixture_rows()
    rows[0]["target"] = "wrong"
    with pytest.raises(ValueError, match="target"):
        summarize_predictions(labels, rows, expected_images=1)
