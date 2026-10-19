from __future__ import annotations

from importlib import import_module

import pytest

torch = pytest.importorskip("torch")
temporal_decision = import_module("tiny_omni_decision.temporal_decision")
CausalGatedTemporalDecision = temporal_decision.CausalGatedTemporalDecision
probability_json = temporal_decision.probability_json
TaskConditionedTemporalDecision = temporal_decision.TaskConditionedTemporalDecision


def test_joint_model_returns_finite_causal_logits_and_gate_probabilities() -> None:
    torch.manual_seed(17)
    model = CausalGatedTemporalDecision(hidden_size=16).eval()
    video = torch.randn(2, 5, 768)
    audio = torch.randn(2, 5, 384)
    text = torch.randn(2, 384)

    result = model(video, audio, text)

    assert result.logits.shape == (2, 5, 2)
    assert result.modality_weights.shape == (2, 5, 2)
    assert torch.isfinite(result.logits).all()
    assert torch.allclose(
        result.modality_weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6
    )
    assert model.trainable_parameter_count < 1_000_000


def test_lstm_is_causal_and_carried_state_matches_full_sequence() -> None:
    torch.manual_seed(23)
    model = CausalGatedTemporalDecision(hidden_size=16).eval()
    video = torch.randn(1, 6, 768)
    audio = torch.randn(1, 6, 384)
    text = torch.randn(1, 384)
    full = model(video, audio, text)

    changed_future = video.clone()
    changed_future[:, 3:] += 100
    changed = model(changed_future, audio, text)
    assert torch.equal(full.logits[:, :3], changed.logits[:, :3])

    first = model(video[:, :3], audio[:, :3], text)
    second = model(video[:, 3:], audio[:, 3:], text, state=first.state)
    assert torch.allclose(full.logits[:, :3], first.logits, atol=1e-6)
    assert torch.allclose(full.logits[:, 3:], second.logits, atol=1e-6)


def test_missing_modality_is_zero_gated_and_single_mode_ignores_it() -> None:
    torch.manual_seed(31)
    model = CausalGatedTemporalDecision(hidden_size=16).eval()
    video = torch.randn(1, 4, 768)
    audio = torch.randn(1, 4, 384)
    text = torch.randn(1, 384)
    video_only = CausalGatedTemporalDecision(hidden_size=16, modality_mode="video_only").eval()
    video_only.load_state_dict(model.state_dict())
    baseline = video_only(video, audio, text)
    altered = video_only(video, audio + 100, text)
    assert torch.equal(baseline.logits, altered.logits)
    assert torch.equal(baseline.modality_weights[..., 1], torch.zeros(1, 4))

    present = torch.tensor([[True, False, True, True]])
    result = model(video, audio, text, audio_present=present)
    assert torch.equal(result.modality_weights[:, 1, 1], torch.zeros(1))
    assert torch.equal(result.modality_weights[:, 1, 0], torch.ones(1))


def test_variable_lengths_mask_padding_from_outputs() -> None:
    model = CausalGatedTemporalDecision(hidden_size=16).eval()
    result = model(
        torch.randn(2, 5, 768),
        torch.randn(2, 5, 384),
        torch.randn(2, 384),
        lengths=torch.tensor([5, 3]),
    )
    assert torch.equal(result.logits[1, 3:], torch.zeros(2, 2))
    assert torch.equal(result.modality_weights[1, 3:], torch.zeros(2, 2))


def test_probability_json_returns_normalized_probabilities_over_options() -> None:
    payload = probability_json(
        torch.tensor([0.0, 1.0]),
        ("No incorrect action", "At least one incorrect action"),
        "session-1:window-0000",
    )
    assert payload["prediction"] == "At least one incorrect action"
    assert [item["option"] for item in payload["probabilities"]] == [
        "No incorrect action",
        "At least one incorrect action",
    ]
    assert sum(item["probability"] for item in payload["probabilities"]) == pytest.approx(1.0)


def test_typed_task_conditioned_readout_has_no_text_input_and_is_causal() -> None:
    torch.manual_seed(37)
    model = TaskConditionedTemporalDecision(task_count=3, hidden_size=16).eval()
    video = torch.randn(2, 5, 768)
    audio = torch.randn(2, 5, 384)
    task_ids = torch.tensor([0, 2])
    result = model(video, audio, task_ids, lengths=torch.tensor([5, 3]))
    assert result.logits.shape == (2, 5, 2)
    assert torch.equal(result.logits[1, 3:], torch.zeros(2, 2))
    assert torch.isfinite(result.logits).all()
    assert not any("text" in name for name, _ in model.named_parameters())
    assert model.trainable_parameter_count < 1_000_000


@pytest.mark.parametrize(
    ("mode", "changed_modality"),
    [("video_only", "audio"), ("audio_only", "video")],
)
def test_task_conditioned_single_modality_arm_ignores_other_sensor(
    mode: str, changed_modality: str
) -> None:
    model = TaskConditionedTemporalDecision(
        task_count=2, hidden_size=16, modality_mode=mode
    ).eval()
    video = torch.randn(1, 4, 768)
    audio = torch.randn(1, 4, 384)
    task_ids = torch.tensor([1])
    baseline = model(video, audio, task_ids)
    if changed_modality == "audio":
        changed = model(video, audio + 100, task_ids)
        ignored_gate = baseline.modality_weights[..., 1]
    else:
        changed = model(video + 100, audio, task_ids)
        ignored_gate = baseline.modality_weights[..., 0]
    assert torch.equal(baseline.logits, changed.logits)
    assert torch.equal(ignored_gate, torch.zeros_like(ignored_gate))


def test_task_conditioned_readout_rejects_unknown_task_ids():
    model = TaskConditionedTemporalDecision(task_count=2, hidden_size=16)
    with pytest.raises(ValueError, match="shape/value is invalid"):
        model(torch.randn(1, 2, 768), torch.randn(1, 2, 384), torch.tensor([2]))
