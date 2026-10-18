# Perception Test AV correspondence task feasibility — 2026-10-18

## Result

A train-only, annotation-level preflight found **2,551 candidate sound events in
1,129 official TRAIN videos** under the fixed rule below. This establishes that
the pinned annotations contain a nontrivial candidate pool for a task asking
which visible object track produced a sound. It is not a model result, task
admission, or evidence that a model needs both modalities.

The source is the official Perception Test revision
`3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`; its official README says software
is Apache-2.0 and other materials are CC-BY-4.0. The annotation schema links
sound segments to object-track IDs, labels, time intervals and visibility.
The upstream paper describes filming-script and participant grouping, but the
inspected public annotation metadata has no participant/script identifiers.
Keep the official split; participant-disjoint generalization remains UNKNOWN.

## Predeclared feasibility rule

An event is counted as a candidate only when it has one resolved parent object,
its label is not `Muted`, the event is marked visible, and the annotated sound
interval ends before any official cut frame. For each event, select the
annotated in-interval frame with at least two visible, nonempty object boxes
closest to the interval midpoint; ties select the earlier frame. This choice
uses interval and visible-track geometry, not the source parent label. If the
parent is absent at that one selected frame, hold the event rather than search
for a more favorable frame.

The resulting task is: given an audio event interval and a supplied set of
object-track image options from the selected frame, return a probability over
the options for which object produced the sound. Candidate counts vary from
**2 to 48** and are not truncated. These are oracle ground-truth track
proposals, so this tests conditional AV source selection, not autonomous object
detection or full-scene perception. The exact candidate-option count
distribution is saved in `result.json`.

The source parent ID is supervision only; it must never be a model input or
option name. Object-crop encoding and a no-answer/abstention contract still
need implementation. All arms must use the same candidate options and compare
audio-only, video-only, joint, shuffled-audio, shuffled-video and missing-input
controls.

All **25,757** sound-segment records are retained in the machine-readable audit.
The 2,551 candidates and 23,206 held cases partition the source annotations;
held cases are not converted into negatives or silently discarded. Reasons:

| Status / reason | Events |
|---|---:|
| Candidate | 2,551 |
| Unknown/unresolved parent | 8,199 |
| Muted audio label | 1,149 |
| Multiple parent sources | 12,444 |
| Source not visible | 230 |
| Reaches/crosses official cut | 26 |
| No simultaneous competing tracks in event | 1,067 |
| Source absent at fixed label-blind frame | 91 |

This candidate pool is narrowly single-source and visible at the selected frame
by construction. Its
sound-label mix is highly skewed; report per-class and per-video support and
query-only/audio-only/video-only controls before considering any fitted head.
The rule has only been exercised on TRAIN annotations. Do not infer validation
coverage, actor/site independence, physical A/V synchronization, or performance
from these counts. The new task is **not admitted to training** yet.

## Reproducibility

Runner: `scripts/audit_perception_av_correspondence.py`. It accepts only the
pinned TRAIN annotation archive and cut map, refuses reused output directories,
preserves every row with a candidate/held status (including `Muted` holds), and
has no model or GPU path.

```powershell
$env:PYTHONPATH='src'
python -m scripts.audit_perception_av_correspondence `
  --source C:/CodexArtifacts/pretrained-reuse-r2/perception-test-source-audit-01 `
  --output C:/CodexArtifacts/pretrained-reuse-r2/perception-av-correspondence-feasibility-08
```

Source SHA-256 values:

- `train_annotations.zip`: `9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78`
- `cut_frame_mapping_train.json`: `cbea6c721fd150f861ec6fe8e7ff005792f754df1359b1364349b5abcaab6cc4`

Attempt 08: `C:/CodexArtifacts/pretrained-reuse-r2/perception-av-correspondence-feasibility-08/`.
`result.json` SHA-256:
`687be9cc6a1575227c73296e6e6677db626968e8c5ba9501d4b3d10e7b4f3f88`.
`records.jsonl` SHA-256:
`31d2cb242845abb2242257430f81563b2e70e2dacf8d877ab4fa6668f2dccbbd`.
The receipt pins its exact runner SHA and records that validation/test media,
model weights, GPU and optimizer were not used. Attempts 01–07 remain preserved;
attempt 08 is the canonical receipt with the fixed frame policy and `Muted`
hold. Its runner SHA-256 is
`9587b9a8c3c5adb3730354eb5e653c780b3ff1b6be9c381d0b3f36851e9b75d8`.

Tests: ten CPU contract tests cover known/unknown and multi-source labels,
visibility, cut boundaries, simultaneous competing boxes, misaligned track
arrays, the `Muted` hold, and confirmation that frame selection never searches
for a frame containing the target. Ruff passes for the runner and tests.

## Official sources

- [Pinned official repository README](https://github.com/google-deepmind/perception_test/blob/3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf/README.md)
- [Perception Test paper](https://papers.nips.cc/paper_files/paper/2023/file/8540fba4abdc7f9f7a7b1cc6cd60e409-Paper-Datasets_and_Benchmarks.pdf)
- [Paper supplement, annotation and participant details](https://papers.nips.cc/paper_files/paper/2023/file/8540fba4abdc7f9f7a7b1cc6cd60e409-Supplemental-Datasets_and_Benchmarks.pdf)
