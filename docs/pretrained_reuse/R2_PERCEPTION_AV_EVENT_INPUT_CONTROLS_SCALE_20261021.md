# Perception AV event-source input sensitivity — 2026-10-21

## Result and scope

**Execution: PASS. Interpretation: audio alignment affects probability quality,
but useful audio contribution remains unproven.** This is one inference-only,
fixed-checkpoint diagnostic on the 146-event / 58-video development-validation
cohort used in the AV event-scale comparison. It is not independent quality
evidence, a blind evaluation, or a checkpoint-selection signal.

The expanded-fit joint checkpoint was reloaded and reproduced the saved natural
predictions exactly. Zeroing audio produced no clear paired change. Replacing
each event's audio with audio from another video increased NLL by 0.208 (95%
video-bootstrap interval for natural minus shuffled: [-0.386, -0.035]); the
Accuracy change was +4.1 percentage points, with an interval spanning zero.
On the subset with a previous same-video event, using that stale audio also
increased NLL (natural minus stale: -0.182, 95% interval [-0.370, -0.040]),
while Accuracy remained uncertain. This suggests sensitivity to audio/video
alignment in the probability distribution. It does not show that the model
uses the correct audio semantics: zeroing is an out-of-distribution input, and
cross-video replacement can alter unrelated acoustic properties.

## Frozen inputs and procedure

- Selected checkpoint:
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-scale-runs-01\expanded-joint\best.pt`
  SHA-256 `d83cb460cebabfa096ee4f33caadde1a9caeb133d84941a7de796160e73cd238`.
- Training receipt SHA-256:
  `6c31e91a0c5cd77785097e921859b8583a1cafa094ada4d9312d03f9f02d6340`.
- Feature manifest SHA-256:
  `54dd89b9a72ec80e679fb9fa653c79ecfd27d6a6a4aaa8a42120c767b2ca958b`.
- Natural development predictions SHA-256:
  `63255ac293d3638fea41415dd76a7b7ea72a7d3ec1a96a4e51afb4bc31e96c98`.
- Diagnostic script SHA-256:
  `fa8c96c9cb21550c9b51cb7cff9d17710fb7e6365301ef512c3876a80d74c759`.
- New output directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-input-controls-02`.
- Conditions: natural audio; zero audio; deterministic audio from another
  video; and the immediately preceding event's audio within the same video.
  Previous-event control applies to 88 events from 35 videos.
- Seed 17; 10,000 paired bootstrap draws resampling video IDs. No optimizer or
  encoder updates; CPU only; no test or sealed-audit data opened.

## Metrics

| Condition | N | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Natural aligned AV | 146 | 0.3288 | 2.0322 | 0.8100 | 0.1983 |
| Zero audio | 146 | 0.3219 | 2.0708 | 0.8221 | 0.1414 |
| Cross-video shuffled audio | 146 | 0.2877 | 2.2397 | 0.8802 | 0.2269 |
| Previous-event audio (subset) | 88 | 0.2841 | 2.1734 | 0.8737 | 0.2354 |
| Natural on the same previous-event subset | 88 | 0.2955 | 1.9915 | 0.8329 | 0.2171 |

Paired natural-minus-corruption bootstrap intervals:

| Comparison | Accuracy delta (95% CI) | NLL delta (95% CI) |
|---|---:|---:|
| Natural minus zero audio | +0.0068 `[-0.0493, 0.0630]` | -0.0386 `[-0.1772, 0.0915]` |
| Natural minus cross-video shuffle | +0.0411 `[-0.0217, 0.1053]` | -0.2076 `[-0.3863, -0.0350]` |
| Natural minus previous-event audio, matched subset | +0.0114 `[-0.0444, 0.0741]` | -0.1819 `[-0.3695, -0.0404]` |

## Reproduction and limits

The runner verifies hashes for the feature manifest, training result,
checkpoint, and saved natural predictions before scoring. Command:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
python -m scripts.analyze_av_event_input_controls `
  --features C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-scale-fits-01\expanded-features.jsonl `
  --checkpoint C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-scale-runs-01\expanded-joint\best.pt `
  --training-result C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-scale-runs-01\expanded-joint\result.json `
  --natural-predictions C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-scale-runs-01\expanded-joint\development-validation-predictions.jsonl `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-input-controls-02 `
  --seed 17
```

The machine-readable receipt is
`C:\CodexArtifacts\pretrained-reuse-r2\perception-av-event-input-controls-02\result.json`,
SHA-256 `c2361f5b8dfe319c2f36c8c98db4bb1eb37af5ab52230feccaa920e6c16f29cc`.
Its per-condition predictions and audio donor map are saved alongside it.
This reused development cohort has already been scored for scale comparison.
Do not tune on these diagnostics or generalize them beyond this narrow
audio-event-to-visible-object-source task. No checkpoint was changed or
promoted.
