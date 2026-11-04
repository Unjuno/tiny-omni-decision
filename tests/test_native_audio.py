from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchaudio')


def module():
    from tiny_omni_decision import native_audio
    return native_audio


def test_native_16k_samples_are_not_normalized_or_resampled_for_speech():
    pcm = torch.linspace(-.3, .7, 35200)
    speech, rate = module().speech_waveform(pcm, 16000)
    assert rate == 16000
    assert torch.equal(speech, pcm)
    assert len(speech) == 35200


def test_starss_channel_already_selected_then_24k_to_16k():
    pcm = torch.ones(120000)
    speech, rate = module().speech_waveform(pcm, 24000)
    assert rate == 16000
    assert speech.shape == (80000,)
    assert torch.isfinite(speech).all()


@pytest.mark.parametrize('pcm,rate', [(torch.zeros(2, 10), 16000),
    (torch.zeros(0), 16000), (torch.zeros(480001), 16000),
    (torch.zeros(2), 8000), (torch.tensor([float('nan')]), 16000),
    (torch.zeros(2, dtype=torch.float64), 16000)])
def test_unsupported_waveforms_fail_without_silent_cropping(pcm, rate):
    with pytest.raises(ValueError):
        module().speech_waveform(pcm, rate)


class Frozen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)


class Speech(Frozen):
    def forward(self, features):
        return SimpleNamespace(last_hidden_state=torch.ones(1, 1500, 384))


class Event(Frozen):
    def forward(self, features):
        return torch.zeros(1, 527), torch.ones(1, 384) * 2


class Mel(Frozen):
    def forward(self, samples):
        self.sample_count = samples.shape[-1]
        return torch.zeros(1, 128, 100)


class Processor:
    sampling_rate = 16000
    n_samples = 480000

    def __call__(self, samples, sampling_rate, return_tensors):
        assert sampling_rate == 16000
        self.sample_count = len(samples)
        return SimpleNamespace(input_features=torch.zeros(1, 80, 3000))


def test_two_audio_paths_keep_physical_speech_tokens_and_full_event_waveform():
    processor, mel = Processor(), Mel().eval()
    speech, events = module().audio_tokens(Speech().eval(), processor, Event().eval(), mel,
                                           torch.zeros(35200), 16000)
    assert speech.shape == (110, 384)
    assert events.shape == (1, 384)
    assert processor.sample_count == 35200
    assert mel.sample_count == 70400


def test_padding_does_not_turn_a_partial_physical_frame_into_30_seconds():
    speech, _ = module().audio_tokens(Speech().eval(), Processor(), Event().eval(), Mel().eval(),
                                      torch.zeros(321), 16000)
    assert speech.shape == (2, 384)


def test_trainable_encoder_is_rejected():
    speech = Speech().eval()
    speech.anchor.requires_grad_(True)
    with pytest.raises(ValueError):
        module().audio_tokens(speech, Processor(), Event().eval(), Mel().eval(),
                              torch.zeros(35200), 16000)
