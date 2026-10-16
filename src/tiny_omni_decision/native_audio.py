"""One frozen speech/event frontend for admitted mono16k/24k source windows."""

import math

import torch
import torchaudio


def speech_waveform(pcm, sample_rate):
    if (not isinstance(pcm, torch.Tensor) or pcm.ndim != 1 or pcm.dtype != torch.float32
            or pcm.device.type != 'cpu' or pcm.requires_grad or type(sample_rate) is not int
            or sample_rate not in {16000, 24000} or not 0 < len(pcm) <= 30 * sample_rate
            or not torch.isfinite(pcm).all()):
        raise ValueError('complete finite CPU FP32 mono16k/24k waveform <=30s required')
    speech = (pcm if sample_rate == 16000 else
              torchaudio.functional.resample(pcm, sample_rate, 16000))
    return speech, 16000


@torch.inference_mode()
def audio_tokens(speech, processor, event, mel, pcm, sample_rate):
    """No transcript, labels, waveform normalization, channel inference or cropping."""
    speech_pcm, speech_rate = speech_waveform(pcm, sample_rate)
    if processor.sampling_rate != 16000 or processor.n_samples != 480000:
        raise ValueError('original30s Whisper processor required')
    device = next(speech.parameters()).device
    for model in (speech, event, mel):
        if (any(m.training for m in model.modules())
                or any(p.requires_grad or p.device != device for p in model.parameters())
                or any(b.device != device for b in model.buffers())):
            raise ValueError('frozen evaluation audio models on one device required')
    features = processor(speech_pcm.numpy(), sampling_rate=speech_rate,
                         return_tensors='pt').input_features
    acoustic = speech(features.to(device)).last_hidden_state
    # Each path starts from the original selected waveform: no24->16->32 two-hop conversion.
    event_pcm = torchaudio.functional.resample(pcm[None].to(device), sample_rate, 32000)
    _, events = event(mel(event_pcm).unsqueeze(1))
    valid = math.ceil(len(speech_pcm) / 320)
    if (acoustic.shape != (1, 1500, 384) or events.shape != (1, 384)
            or not torch.isfinite(acoustic).all() or not torch.isfinite(events).all()):
        raise ValueError('finite original speech/event representations required')
    return acoustic[0, :valid], events
