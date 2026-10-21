"""Explicit event-window positions and direct prepared audio; no old frontend changes."""

import math

import numpy as np


def event_positions(video_pts, start, end):
    pts = np.asarray(video_pts)
    if (pts.shape != (8,) or not np.isfinite(pts).all() or not (np.diff(pts) > 0).all()
            or not all(math.isfinite(t) for t in (start, end))
            or not math.isclose(end - start, .5, abs_tol=1e-9, rel_tol=0)
            or not ((pts >= start) & (pts < end)).all()):
        raise ValueError('eight actual increasing PTS inside a half-second window required')
    time = (pts - start) / (end - start)
    grid = np.stack(np.meshgrid(time, np.linspace(0, 1, 16), np.linspace(0, 1, 16),
                               indexing='ij'), axis=-1).reshape(2048, 3)
    speech = np.zeros((25, 3), np.float32)
    speech[:, 0] = np.arange(25) * .02 / (end - start)
    return dict(video=grid.astype(np.float32), speech=speech,
                event_audio=np.array([[.5, 0., 0.]], dtype=np.float32))


def prepared_audio_tokens(speech, processor, event, mel, pcm16, pcm32):
    for values, length in ((pcm16, 8000), (pcm32, 16000)):
        if (not isinstance(values, np.ndarray) or values.shape != (length,)
                or values.dtype != np.float32 or not np.isfinite(values).all()):
            raise ValueError('finite exact half-second float32 prepared audio required')
    import torch

    if processor.sampling_rate != 16000 or processor.n_samples != 480000:
        raise ValueError('original Whisper30s processor required')
    device = next(speech.parameters()).device
    for model in (speech, event, mel):
        if (any(m.training for m in model.modules())
                or any(p.requires_grad or p.device != device or p.dtype != torch.float32
                       for p in model.parameters())
                or any(b.device != device for b in model.buffers())):
            raise ValueError('frozen evaluation FP32 models on one device required')
    with torch.inference_mode():
        features = processor(pcm16.copy(), sampling_rate=16000,
                             return_tensors='pt').input_features
        acoustic = speech(features.to(device)).last_hidden_state
        _, events = event(mel(torch.from_numpy(pcm32.copy())[None].to(device)).unsqueeze(1))
        if (acoustic.shape != (1, 1500, 384) or events.shape != (1, 384)
                or not torch.isfinite(acoustic).all() or not torch.isfinite(events).all()):
            raise ValueError('finite original audio feature shapes required')
        return acoustic[0, :25], events
