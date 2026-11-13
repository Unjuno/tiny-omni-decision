"""Interventions at an immutable temporal head's visual/readout boundary."""

import numpy as np
import torch


def _finite(value, shape):
    if (
        value.shape != shape
        or value.dtype != torch.float32
        or value.device.type != "cpu"
        or not torch.isfinite(value).all()
    ):
        raise ValueError("finite CPU FP32 boundary dimensions required")


def _mask(value):
    _finite(value, (len(value), 2))
    if not len(value) or not ((value == 0) | (value == 1)).all():
        raise ValueError("nonempty binary presence mask required")


def latent_states(model, video):
    _finite(video, (len(video), 16, 768))
    if not len(video):
        raise ValueError("nonempty native visual sequence required")
    model.eval()
    with torch.no_grad():
        result = torch.cat(
            [
                model.temporal(model.visual(video[i : i + 128]))[0][:, -1]
                for i in range(0, len(video), 128)
            ]
        )
    _finite(result, (len(video), 64))
    return result


def readout_from_latent(model, audio, latent, present):
    count = len(audio)
    _finite(audio, (count, 384))
    _finite(latent, (count, 64))
    _finite(present, (count, 2))
    _mask(present)
    model.eval()
    with torch.no_grad():
        return np.concatenate(
            [
                model.readout(
                    torch.cat(
                        (
                            audio[i : i + 128] * present[i : i + 128, :1],
                            latent[i : i + 128] * present[i : i + 128, 1:2],
                            present[i : i + 128],
                        ),
                        -1,
                    )
                ).numpy()
                for i in range(0, count, 128)
            ]
        )


def latent_controls(train, present):
    _finite(train, (len(train), 64))
    _mask(present)
    if not len(train):
        raise ValueError("training-only visual states required")
    count = len(present)
    mean = torch.from_numpy(train.numpy().mean(0, dtype=np.float64).astype(np.float32))
    mask = present.clone()
    mask[:, 1] = 0
    return dict(
        latent_train_mean=(mean.repeat(count, 1), present.clone()),
        latent_train_first=(train[:1].repeat(count, 1), present.clone()),
        latent_zero_present=(torch.zeros(count, 64), present.clone()),
        latent_zero_absent=(torch.zeros(count, 64), mask),
    )


def summary(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not len(value) or not np.isfinite(value).all():
        raise ValueError("finite row/channel matrix required")
    std = value.std(0)
    return dict(
        rows=len(value),
        channels=value.shape[1],
        std_mean=float(std.mean()),
        std_max=float(std.max()),
        std_min=float(std.min()),
        std_below_1e6=int((std < 1e-6).sum()),
        std_below_1e3=int((std < 1e-3).sum()),
        fraction_abs_at_least_099=float((abs(value) >= 0.99).mean()),
    )
