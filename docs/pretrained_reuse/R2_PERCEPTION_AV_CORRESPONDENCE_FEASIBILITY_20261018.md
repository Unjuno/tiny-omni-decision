# Perception Test AV correspondence task feasibility — 2026-10-18

## Result

A train-only, annotation-level preflight found **2,594 candidate sound events in
1,139 official TRAIN videos** under the fixed rule below. This establishes that
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
interval ends before any official cut frame. At a shared annotated frame within
the event, both the source and at least one other unmasked object must have
nonempty boxes. This creates
a plausible candidate-choice task: compare a sound interval with supplied
visual object options and return a probability over the options.

The audit does **not** choose one frame or define how candidate object options
will be encoded. Choosing a frozen frame, deciding whether choices are object
crops or track clips, and specifying the no-answer/abstention case still require
a protocol. The source parent ID is a supervision label only; it must never be
passed as a model input or option name. The eventual protocol must use the same
complete candidate-generation rule in all arms and compare audio-only,
video-only, joint, shuffled-audio, shuffled-video and missing-input controls.

All **25,757** sound-segment records are retained in the machine-readable audit.
The 2,594 candidates and 23,163 held cases partition the source annotations;
held cases are not converted into negatives or silently discarded. Reasons:

| Status / reason | Events |
|---|---:|
| Candidate | 2,594 |
| Unknown/unresolved parent | 8,199 |
| Muted audio label | 1,149 |
| Multiple parent sources | 12,444 |
| Source not visible | 230 |
| Reaches/crosses official cut | 26 |
| No shared annotated frame with a visible competitor | 1,115 |

This candidate pool is narrowly single-source and visible by construction. Its
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
  --output C:/CodexArtifacts/pretrained-reuse-r2/perception-av-correspondence-feasibility-05
```

Source SHA-256 values:

- `train_annotations.zip`: `9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78`
- `cut_frame_mapping_train.json`: `cbea6c721fd150f861ec6fe8e7ff005792f754df1359b1364349b5abcaab6cc4`

Attempt 05: `C:/CodexArtifacts/pretrained-reuse-r2/perception-av-correspondence-feasibility-05/`.
`result.json` SHA-256:
`2cd15b09fb25b5af959584c366848478e9e7a08b774ef3432d90b934e2cf0c35`.
`records.jsonl` SHA-256:
`bb0fb9aa144cd2a768abc2b3a2e14ddfeb1305a9fcf2007a2d7f651448b6236c`.
The receipt pins its exact runner SHA and records that validation/test media,
model weights, GPU and optimizer were not used. Attempts 01–03 were preserved;
attempts 04 and 05 are both preserved; attempt 05 reflects the added `Muted`
hold rule and is the result corresponding to the final runner.

Tests: nine CPU contract tests cover known/unknown and multi-source labels,
visibility, cut boundaries, simultaneous competing boxes, misaligned track
arrays, the `Muted` hold, and confirmation that objects annotated at different
times do not count as simultaneous options. Ruff passes for the runner and tests.

## Official sources

- [Pinned official repository README](https://github.com/google-deepmind/perception_test/blob/3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf/README.md)
- [Perception Test paper](https://papers.nips.cc/paper_files/paper/2023/file/8540fba4abdc7f9f7a7b1cc6cd60e409-Paper-Datasets_and_Benchmarks.pdf)
- [Paper supplement, annotation and participant details](https://papers.nips.cc/paper_files/paper/2023/file/8540fba4abdc7f9f7a7b1cc6cd60e409-Supplemental-Datasets_and_Benchmarks.pdf)
