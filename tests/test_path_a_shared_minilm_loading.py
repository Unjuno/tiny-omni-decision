from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch", reason="optional Path A runtime dependency")
pytest.importorskip("transformers", reason="optional Path A runtime dependency")
pytest.importorskip("psutil", reason="optional Path A runtime dependency")

from scripts import path_a_torch_worker  # noqa: E402
from tiny_omni_decision.audio_runtime import SpeechCommandsMultiQueryRuntime  # noqa: E402
from tiny_omni_decision.text_runtime import TypedTextDecisionRuntime  # noqa: E402
from tiny_omni_decision.video_runtime import ClevrerBinaryVideoFeatureRuntime  # noqa: E402


class _FakeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


def test_torch_worker_loads_minilm_once_and_injects_same_instances(monkeypatch, tmp_path) -> None:
    encoder = _FakeEncoder()
    tokenizer = object()
    model_loader = Mock(return_value=encoder)
    tokenizer_loader = Mock(return_value=tokenizer)
    monkeypatch.setattr(
        path_a_torch_worker,
        "AutoModel",
        SimpleNamespace(from_pretrained=model_loader),
    )
    monkeypatch.setattr(
        path_a_torch_worker,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=tokenizer_loader),
    )

    audio = SimpleNamespace(
        audio_model=torch.nn.Linear(2, 2),
        text_model=encoder,
        text_tokenizer=tokenizer,
        scorer=torch.nn.Linear(2, 2),
    )
    text = SimpleNamespace(
        encoder=encoder,
        tokenizer=tokenizer,
        scorer=torch.nn.Linear(2, 2),
    )
    video = SimpleNamespace(
        encoder=encoder,
        tokenizer=tokenizer,
        scorer=torch.nn.Linear(2, 2),
    )
    audio_factory = Mock(return_value=audio)
    text_factory = Mock(return_value=text)
    video_factory = Mock(return_value=video)
    monkeypatch.setattr(path_a_torch_worker, "SpeechCommandsMultiQueryRuntime", audio_factory)
    monkeypatch.setattr(path_a_torch_worker, "TypedTextDecisionRuntime", text_factory)
    monkeypatch.setattr(path_a_torch_worker, "ClevrerBinaryVideoFeatureRuntime", video_factory)

    actual = path_a_torch_worker._build_task_runtimes(
        tmp_path, device="cpu", audio_cache_entries=64
    )

    assert actual == (audio, text, video)
    model_loader.assert_called_once_with(
        tmp_path / "components/audio/models/text-minilm-en",
        local_files_only=True,
        dtype=torch.float32,
    )
    tokenizer_loader.assert_called_once_with(
        tmp_path / "components/audio/models/text-minilm-en", local_files_only=True
    )
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    modules = [
        audio.audio_model,
        audio.text_model,
        audio.scorer,
        text.encoder,
        text.scorer,
        video.scorer,
    ]
    assert path_a_torch_worker._unique_parameters(modules) == sum(
        parameter.numel()
        for module in (audio.audio_model, encoder, audio.scorer, text.scorer, video.scorer)
        for parameter in module.parameters()
    )
    for factory in (audio_factory, text_factory, video_factory):
        kwargs = factory.call_args.kwargs
        assert kwargs.get("shared_encoder", kwargs.get("shared_text_encoder")) is encoder
        assert kwargs.get("shared_tokenizer", kwargs.get("shared_text_tokenizer")) is tokenizer


@pytest.mark.parametrize(
    ("runtime", "encoder_argument"),
    [
        (SpeechCommandsMultiQueryRuntime, "shared_text_encoder"),
        (TypedTextDecisionRuntime, "shared_encoder"),
        (ClevrerBinaryVideoFeatureRuntime, "shared_encoder"),
    ],
)
def test_shared_encoder_requires_its_matching_tokenizer(
    runtime, encoder_argument, tmp_path
) -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        runtime(tmp_path, tmp_path, **{encoder_argument: _FakeEncoder()})
