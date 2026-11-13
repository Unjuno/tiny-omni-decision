import pytest
import torch

from scripts.native_gradient_diagnostic import (
    MODALITIES,
    fixed_update_window,
    required_bank_names,
    summarize_updates,
    update_gradient_metrics,
)


def test_fixed_window_selects_the_final_prior_updates():
    order = [list(range(step * 8, (step + 1) * 8)) for step in range(10)]
    start, window = fixed_update_window(order, selected_step=8, window_updates=3)
    assert start == 5
    assert window == order[5:8]


@pytest.mark.parametrize(
    "args",
    [
        ([], 1, 1),
        ([[0] * 7], 1, 1),
        ([[0] * 8], 0, 1),
        ([[0] * 8], 1, 2),
    ],
)
def test_fixed_window_rejects_invalid_ranges_and_batches(args):
    with pytest.raises(ValueError):
        fixed_update_window(*args)


def test_required_banks_include_observation_and_train_query_feature_sources():
    observations = {"image-1": {"features": [{"bank": "image/bank.npy"}]}}
    query_map = {"question-a": {"bank": "language/queries.npy"}}
    assert required_bank_names(observations, {"image-1"}, query_map) == {
        "image/bank.npy",
        "language/queries.npy",
    }


def test_gradient_metrics_separate_shared_direction_and_global_clipping():
    values = {
        "text": torch.tensor([1.0, 0.0]),
        "image": torch.tensor([-1.0, 0.0]),
        "audio": torch.tensor([0.0, 1.0]),
        "video": torch.tensor([0.0, -1.0]),
    }
    shared = {name: values[name] for name in MODALITIES}
    private = {name: torch.ones(2) for name in MODALITIES}
    metrics = update_gradient_metrics(values, shared, private, clip_norm=0.5)
    assert metrics["pairwise_cosine"]["text|image"]["shared_parameters"] == -1.0
    assert metrics["pairwise_cosine"]["text|audio"]["all_parameters"] == 0.0
    assert metrics["gradient_norms"]["global_batch_before_clip"] == 0.0
    assert metrics["global_clip_scale"] == 1.0
    assert metrics["shared_gradient_cancellation_ratio"] == 0.0


def test_gradient_metrics_allow_distinct_parameter_scope_lengths():
    all_parameters = {name: torch.ones(6) for name in MODALITIES}
    shared = {name: torch.ones(4) for name in MODALITIES}
    private = {name: torch.ones(2) for name in MODALITIES}

    metrics = update_gradient_metrics(all_parameters, shared, private, clip_norm=1.0)

    assert metrics["gradient_norms"]["all_parameters_by_modality"]["text"] == pytest.approx(
        6**0.5
    )
    assert metrics["gradient_norms"]["shared_parameters_by_modality"]["text"] == 2.0
    assert metrics["gradient_norms"]["modality_projection_parameters_by_modality"][
        "text"
    ] == pytest.approx(2**0.5)


def test_gradient_metrics_reject_nonfinite_or_misaligned_vectors():
    values = {name: torch.ones(2) for name in MODALITIES}
    bad = dict(values)
    bad["video"] = torch.ones(3)
    with pytest.raises(ValueError):
        update_gradient_metrics(values, bad, values, clip_norm=1.0)
    bad = dict(values)
    bad["video"] = torch.tensor([float("nan"), 0.0])
    with pytest.raises(ValueError):
        update_gradient_metrics(values, values, bad, clip_norm=1.0)


def test_update_summary_reports_negative_cosine_and_clipping_frequency():
    base = {
        "pairwise_cosine": {"text|image": {"all_parameters": -0.5, "shared_parameters": -0.25}},
        "global_clip_scale": 0.5,
        "shared_gradient_cancellation_ratio": 0.25,
    }
    summary = summarize_updates([base, base | {"global_clip_scale": 1.0}])
    assert summary["updates"] == 2
    assert summary["pairwise_cosine"]["text|image"]["shared_parameters"]["negative_fraction"] == 1.0
    assert summary["fraction_updates_clipped"] == 0.5
