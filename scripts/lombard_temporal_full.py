"""Admission and read-only evidence checks for the fixed native temporal experiment."""

import json

import numpy as np
import torch

from scripts.lombard_readout_core import summarize
from scripts.train_lombard_temporal import ARMS, admit_smoke


def admit_complete_smoke(reference, pins):
    admit_smoke(reference, pins)
    expected = dict(
        status="PASS_TRAIN_ONLY_SMOKE",
        updates=2,
        train_unique_assets=4,
        consumed_assets=8,
        head_parameters=106355,
        final_reload_exact=True,
        json_option_permutation_exact=True,
    )
    for arm in ARMS:
        if any(reference["arms"][arm].get(k) != v for k, v in expected.items()):
            raise ValueError("complete four-train-only two-update smoke required for every arm")
    for key in ["initial_tensor_sha256", "sampler_sha256"]:
        if len({reference["arms"][a][key] for a in ARMS}) != 1:
            raise ValueError("smoke initialization/sample order differs")


def controls(values, permutation):
    audio, video, present = values
    p = torch.as_tensor(permutation, dtype=torch.long)
    if sorted(p.tolist()) != list(range(len(audio))):
        raise ValueError("control donors must be an exact bijection")
    audio_mask, video_mask = present.clone(), present.clone()
    audio_mask[:, 0] = 0
    video_mask[:, 1] = 0
    return dict(
        matched=values,
        reverse=(audio, video.flip(1), present),
        audio_mismatch=(audio[p], video, present),
        video_mismatch=(audio, video[p], present),
        audio_missing=(torch.zeros_like(audio), video, audio_mask),
        video_missing=(audio, torch.zeros_like(video), video_mask),
    )


def verify_snapshot(path, rows, targets, metrics=None, temperatures=None, replayed_logits=None):
    with np.load(path, allow_pickle=False) as saved:
        logits = saved["logits"].copy()
        actual, probabilities = summarize(logits, targets, temperatures)
        if (
            saved["source_ids"].tolist() != [r["source_id"] for r in rows]
            or saved["speaker_groups"].tolist() != [r["speaker_group"] for r in rows]
            or not np.array_equal(saved["targets"], targets)
            or not np.array_equal(saved["probabilities"], probabilities)
        ):
            raise ValueError("snapshot identities/targets/probabilities differ")
    if metrics is not None and actual != metrics:
        raise ValueError("saved metrics differ from full prediction snapshot")
    if replayed_logits is not None and not np.array_equal(logits, replayed_logits):
        raise ValueError("snapshot raw logits differ from freshly replayed checkpoint")
    return logits


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))
