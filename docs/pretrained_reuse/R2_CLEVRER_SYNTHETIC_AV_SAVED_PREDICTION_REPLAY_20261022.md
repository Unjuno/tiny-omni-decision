# CLEVRER synthetic AV v2 saved-prediction replay (2026-10-22)

## Purpose and scope

This is an independent CPU replay of the selected joint and audio-only
predictions from the PR #21 synthetic CLEVRER audiovisual coverage run. It
does not load model weights, checkpoints, feature caches, or media, and does
not train, select, or evaluate a new candidate. It preserves the existing
experiment and recomputes metrics from its saved prediction probabilities.

The replay implementation is
[`scripts/replay_saved_synthetic_av_predictions.py`](../../scripts/replay_saved_synthetic_av_predictions.py).
It checks row alignment, unique IDs, binary normalized probabilities, and
prediction/probability agreement before reporting Accuracy, NLL, Brier, 15-bin
ECE, and mean confidence. It also recomputes the paired scene bootstrap and
audio-swap intervention diagnostics.

## Reproduction

```powershell
python scripts/replay_saved_synthetic_av_predictions.py `
  --run-dir C:\CodexArtifacts\pretrained-reuse-r2\clevrer-synthetic-av-temporal-decision-v2-coverage-attempt-01 `
  --repetitions 5000 --seed 17
```

The source run is PR #21, branch commit
`61b02dbbb84102f87bcaaaec6ecb54ff935a7705`. Its artifact directory is outside
Git. The replay observed 1,110 validation rows from 240 scenes and 555 paired
events. Input hashes:

| File | SHA-256 |
| --- | --- |
| `predictions-best-joint_lstm.jsonl` | `a2ab7a1a1f27fce49ffc3233ed81fd186f8eb7f788fc355105a33b15e8c6f587` |
| `predictions-best-audio_only_lstm.jsonl` | `be07ce9d6be7f3d2e3ad0122d95e242a5e1bc699405cadbd09acaebed16b4b26` |
| `predictions-best-joint_lstm-audio-paired-swap.jsonl` | `bf0fea8d1b3174219593245b503c70fd6a868462bb8cc152cebf5190a86cd28b` |
| `results.json` | `3d9545e9fc5b9abaccffbc463ba1f2d3f5a2790d0ed3cfa37b8a4341310cc8ff` |
| `run-metadata.json` | `fb41265749f7405e74b26ca108e533717201b728b362f8e994cc2d79938ae8fc` |
| `effective-config.json` | `75f690018eaf912113bbf14d3bcbf66de150033abba11af41a2892858414daaa` |

## Recomputed results

| Readout | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Joint gated LSTM | 0.835135 | 0.353379 | 0.224368 | 0.036942 | 0.867555 |
| Audio-only LSTM | 0.751351 | 0.457741 | 0.319263 | 0.021328 | 0.752481 |

Paired bootstrap resampled 240 scenes 5,000 times with seed 17; each replicate
sampled scenes and retained all examples within a sampled scene. Joint minus
audio-only Accuracy was `+0.083507` (95% percentile interval
`[0.058854, 0.108338]`). Audio-only minus joint NLL was `0.118521`
(`[0.084078, 0.151214]`), and Brier improvement was `0.103548`
(`[0.083436, 0.124232]`). These are development-validation diagnostics and
inherit checkpoint-selection/validation reuse; they are not blind estimates.

After swapping paired audio, accuracy on the original target was `0.164865`,
accuracy on the opposite donor target was `0.835135`, and predictions changed
on `0.699099` of rows. This establishes sensitivity to the paired synthetic
audio intervention in these saved predictions, not broad natural AV
understanding or robust complementary reasoning.

The recomputed values match the source report/run results at displayed
precision. The original result remains a synthetic cue experiment: CLEVRER
itself has no audio, and the generated tone is aligned around a visual
collision. The validation also has known scene/task and question-template
exposure limitations documented in the PR #21 v2 report. It is not a sealed
audit and does not satisfy the R2 natural audiovisual gate or the product
quality target.

## Verification

- `python -m pytest -q tests/test_replay_saved_synthetic_av_predictions.py`: 5 passed.
- `ruff check scripts/replay_saved_synthetic_av_predictions.py tests/test_replay_saved_synthetic_av_predictions.py`: passed.
- No GPU, model, checkpoint, or media was accessed for this replay.
