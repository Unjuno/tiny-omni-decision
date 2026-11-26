# Perception Test fixed-readout train-transfer diagnostic

Date: 2026-10-21. Status: the fixed-checkpoint diagnostic completed. This is a
small, source-train-only transfer check, not independent validation, blind
evaluation, or a product result.

## Question and fixed design

The existing seed-17 audio-only, video-only, and joint linear readouts were
applied without retraining to one new event from each of 256 Perception Test
official-training videos. These video IDs are disjoint from the 256 official
training video IDs used to fit the readout. The cohort is balanced by target
(128 not-visible, 128 visible), selected by the frozen deterministic SHA-256
rule. Annotation labels were read to form that balanced cohort, so the result is
nonblind and remains within the training source distribution.

Each event uses the same centered two-second audio/video interval as the fitted
readout: eight uniformly spaced video target times, nearest decoded frame PTS,
and the fixed EfficientAT and VideoPrism checkpoints. All three saved readout
heads were scored exactly once on the same 256 feature pairs. No weights,
normalization statistics, thresholds, data composition, or checkpoints were
updated or selected. No validation/test media, labels, or sealed-audit data were
opened in this attempt. The evaluator reads only the original fit cohort's
`train` IDs to verify group disjointness.

## Integrity correction and attempts

Attempt 01 used a 250 ms event-to-media boundary margin. It was frozen before
media acquisition, but that margin did not guarantee the readout's full
centered two-second feature window. Video extraction safely stopped at the
first row, `video_360`, whose requested interval began at -0.4151 s and whose
nearest-frame error was 0.4151 s. No model was evaluated. The cohort, acquired
media, and partial `run.json` were retained unchanged at:

`C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-01`

Before any model score was inspected, the technical eligibility rule was
corrected to require a full one-second margin around the event center. Attempt
02 was selected from the same pinned official train annotations with seed 17,
still one event per video and 128 rows per class. It shared 227 video IDs with
attempt 01, which were SHA-256 checked and hard-linked; only the 29 replacement
video/audio pairs were newly acquired. Attempt 01 was not overwritten.

Attempt 02 is at:

`C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02`

The attempt 02 `preflight.json` was written before a field-name cleanup in the
script. Its `expected_retained_media_bytes` field is the **493,278,217
additional** bytes for 29 new pairs, not the full cohort's 4,932,782,217
logical media bytes. The checked-in script now calls this
`expected_additional_media_bytes`.

The selected media set expands to 4,932,782,217 bytes. The added 29 pairs
account for 493,278,217 bytes. All downloaded archive members passed size and
CRC checks; the complete 512-file media index contains per-file SHA-256 values.
The archive ETags and source revision are recorded in the local receipts.

## Metrics

All metrics below use the same balanced, video-disjoint train-transfer cohort.
The reference column is the already-observed original 256-example balanced
official-validation result from the same frozen heads. It is shown only as
context; the two cohorts are different and are not a paired comparison.

| Head | New train-transfer Accuracy | NLL | Brier | ECE | Previously observed validation Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Audio-only | 0.5352 | 1.3657 | 0.3519 | 0.3138 | 0.5625 | 1.1991 | 0.3250 | 0.2744 |
| Video-only | 0.6523 | 1.3021 | 0.2848 | 0.2600 | 0.5781 | 1.4038 | 0.3370 | 0.2927 |
| Joint | 0.6602 | 1.1195 | 0.2579 | 0.2182 | 0.6484 | 1.0884 | 0.2839 | 0.2561 |

New-cohort class recalls (`not-visible`, `visible`) were 0.5000 / 0.5703 for
Audio-only, 0.6328 / 0.6719 for Video-only, and 0.6250 / 0.6953 for Joint.
The joint-minus-audio accuracy difference was +12.50 percentage points; paired
example bootstrap (2,000 resamples, seed 17, percentile 95% interval) was
[+5.47, +19.14] points. Joint minus Video-only was +0.78 points with interval
[-4.30, +5.86], so this cohort does not establish an Accuracy gain over Video
alone. Joint had lower NLL, Brier, and ECE than Video-only on this cohort, but
its ECE remained 0.218 and mean confidence was 0.878. The readouts remain
poorly calibrated.

The validation-to-transfer changes are descriptive only: Audio-only decreased
2.73 points in Accuracy and worsened on NLL/Brier/ECE; Video-only increased
7.42 points and improved on NLL/Brier/ECE; Joint increased 1.17 points, with
slightly worse NLL but better Brier/ECE. They cannot isolate a causal source of
the differences because they are unpaired cohorts from different partitions.

## Interpretation and limits

This is evidence that the frozen VideoPrism-only head transfers to new videos
within the official training split better than the frozen EfficientAT-only
head on this selected balanced subset. It does not prove a population-level
video advantage. Joint accuracy is statistically unresolved against Video-only
in the paired bootstrap; therefore this task does not yet show that audio adds
necessary information to video. Better Joint NLL/Brier/ECE is a useful signal
for a future preregistered, rights-cleared both-required task, not evidence of
robust multimodal fusion.

The sample is 256 groups, selected and class-balanced using labels already
read from the training annotations. It must not be called an independent test,
blind validation, or a 90% product-quality result. No text, still-image, sensor,
or broader audio/video task was evaluated. No new training, checkpoint
selection, quantization, or cloud compute occurred.

## Compute and model preservation

- VideoPrism: 114,365,184 frozen parameters; pinned checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`; CPU;
  611.23 seconds for 256 examples; maximum frame-time error 17.72 ms.
- EfficientAT: 1,961,239 frozen parameters; pinned checkpoint SHA-256
  `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`; RTX
  3080 Laptop; 10.17 seconds for 256 examples; peak allocated VRAM 43,603,456
  bytes and peak reserved VRAM 67,108,864 bytes. EfficientAT state hashes before
  and after match exactly.
- HotUGPU waited for Orbitune PID 24700 to yield, reserved 3,200 MiB, and
  launched the child with 13,015 MiB free. After the guarded child exited, the
  Orbitune supervisor resumed training as PID 28928. No process was stopped,
  restarted, or otherwise intervened with by this work.
- No cloud compute or cost was used. CPU extraction and GPU feature extraction
  are the only model compute; the readout evaluation itself is CPU-only.
- The existing readout result (`95ee8300…a224`), normalization, and three head
  files were hash-checked before use and left unchanged. All new datasets,
  features, predictions, and receipts are outside Git.

## Artifact hashes

- Source annotation archive: `0200a50644ab9d36a2dd80e5239e709cabeeaf2e4274eca201c90743b89cc6f5`.
- Original fit cohort: `dd8c9de6d8970eb1105cb81551e386fca563eb174b8bcda5b09dcc8a16e567be`.
- Attempt 02 frozen transfer cohort: `910a043363d1d989da411738ea53fa4fa897c4ba59ff05fe71cdc7646df3163e`.
- Attempt 02 inference manifest: `2cb6a492aa671d544f9e102a44619a0c89bf4819e7eabda74168561b87081d4e`.
- Merged 512-file media index: `11ff54ffd14ba15709ed76faf9f29d9b855c1a4686cd16c87930c493c931f77f`.
- VideoPrism features receipt: `d7322c87cb59cf57990179c6446324601a83fb7e40e72a89158d594e5adb758a`.
- EfficientAT features receipt: `930e8ee0615d92028f2b823ff79455b87d85a173b08bb5d5b3d4ae4f2b339c7a`.
- Fixed-readout evaluation and per-example predictions:
  `35c122c255db6d2841f8e87105a4a18701955283b793968cfc515a0fc9bd2524`.
- Fixed normalization: `ddcad9c498cecb43d3faa65c55bd3305e75fc19cf99e7e19c80344ef15ac16f6`.
- Fixed heads: Audio `3ac71d79327f6f992022950e3e2f65791e3a0ac6c4c9f18e2c3d1754feb81bf1`, Video `0f5895627d54143dffc482da96699609fc24164663d65a6af78ef44af1231df5`, Joint `8ae155133858d1b768613a3782d2ce2e5c28f54c2179377ce50ae48588719ee2`.

The evaluation JSON does not embed the evaluator source hash. After this one-time
score was saved, code review added explicit cohort-to-manifest event/target/window
consistency checks without changing the scoring calculations. The current
evaluator source hash is recorded in the PR commit; the run was not repeated to
avoid rescoring the fixed cohort.

## Repository changes and verification

Added deterministic cohort/manifest/media-receipt helpers, fixed-head scoring,
and CPU tests. Verification: Ruff passed for all changed Python files; the full
CPU suite passed (2,327 passed, 11 skipped). As a non-model verification, the
saved original-validation probabilities were used to recompute all three
heads' Accuracy/NLL/Brier/ECE; each difference from the stored metrics was
exactly zero. Existing Teacher v1/v2 artifacts and attempts 01/02 are retained.
No PR was merged.

Current evaluator source SHA-256 (after adding post-run input-consistency
guards):
`28bc2dfd0300f70d3224a1de5548c1cd78455bd1f126e0c8141276574c4b52c2`. The
scoring calculation was unchanged by those post-run guards; the single scoring
run was not repeated. The selected cohort's saved score remains the immutable
`fixed_readout_evaluation.json` hash above.

Before applying it to the new cohort, the head/normalization implementation
was checked against the already-saved original validation probabilities using
the original frozen features. Maximum absolute probability errors were
`9.99e-16` Audio, `8.88e-16` Video, and `1.05e-15` Joint. This confirmed the
Audio-then-Video feature concatenation and train-only normalization without
changing the readout.

## Reproduction commands

Run from the repository root. Commands for later stages use attempt 02; do not
point them at attempt 01.

```powershell
$env:PYTHONPATH = 'src'
python scripts/prepare_perception_visibility_transfer.py `
  --reuse-media-root 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-01\media' `
  --reuse-acquisition 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-01\acquisition.json'

python scripts/build_perception_visibility_transfer_manifest.py `
  --cohort 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\transfer_cohort.json' `
  --output 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\inference_manifest.json'
```

The acquisition manifest was derived without changing cohort rows:

```python
import json
from pathlib import Path

root = Path(r"C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02")
cohort = json.loads((root / "transfer_cohort.json").read_text(encoding="utf-8"))
manifest = {
    "source_revision": cohort["source_revision"],
    "source_split": "official train only",
    "train": cohort["examples"],
}
(root / "acquisition_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
```

```powershell
python -m scripts.acquire_perception_temporal_match_train_media `
  --manifest 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\acquisition_manifest.json' `
  --output 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\media' `
  --existing-media-root 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-01\media' `
  --report 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\acquisition.json'

python scripts/assemble_perception_visibility_transfer_media.py `
  --cohort 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\transfer_cohort.json' `
  --reuse-root 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-01\media' `
  --reuse-receipt 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-01\acquisition.json' `
  --new-root 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\media' `
  --new-receipt 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\acquisition.json'
```

Before extraction, the run creates an empty `empty-fresh-validation.json` with
`{"examples_manifest": [], "validation_media_opened": false}`. VideoPrism
then runs in the pinned CPU environment:

```powershell
$env:PYTHONPATH = 'C:\CodexWorktrees\r0-r2-continuation-audit-20261021\src;C:\CodexWorktrees\r0-r2-continuation-audit-20261021'
& 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe' `
  -m scripts.extract_perception_temporal_match_videoprism `
  --manifest 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\inference_manifest.json' `
  --checkpoint 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-f16r288.npz' `
  --source 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source' `
  --old-media 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\media' `
  --old-feature-root 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-v1\features_videoprism' `
  --added-train-media 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\media' `
  --fresh-valid-media 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\media' `
  --train-acquisition 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\acquisition.json' `
  --fresh-valid-media-report 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\empty-fresh-validation.json' `
  --output 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\features_videoprism'
```

Audio extraction used the cooperative gate, then the CPU-only fixed-head
evaluator:

```powershell
python C:\CodexWorktrees\pretrained-reuse-next-20261021\scripts\run_with_hotgpu_guard.py `
  --max-wait-seconds 7200 --reserve-mib 3200 --block-if-free-mib 14000 --min-child-free-mib 8000 `
  'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-v1\env\Scripts\python.exe' `
  -m scripts.extract_perception_temporal_match_efficientat `
  --manifest 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\inference_manifest.json' `
  --weights 'C:\CodexArtifacts\pretrained-reuse-r1\efficientat-audit-v1\dymn04_as.pt' `
  --source 'C:\CodexArtifacts\pretrained-reuse-r1\efficientat-audit-v1\source' `
  --old-media 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\media' `
  --added-train-media 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\media' `
  --fresh-valid-media 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\media' `
  --train-acquisition 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\acquisition.json' `
  --fresh-valid-media-report 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\empty-fresh-validation.json' `
  --output 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\features_efficientat'

python scripts/evaluate_perception_visibility_transfer.py `
  --cohort 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\transfer_cohort.json' `
  --fit-cohort 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-mcqa-screen-20261021\fixed_av_cohort_v1_seed17.json' `
  --inference-manifest 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\inference_manifest.json' `
  --audio-result 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\features_efficientat\result.json' `
  --video-result 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\features_videoprism\result.json' `
  --readout-dir 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-v1\matched_linear_readout_v1' `
  --output 'C:\CodexArtifacts\pretrained-reuse-r2\perception-test-av-train-transfer-20261021-02\fixed_readout_evaluation.json'
```

The commands are included for audit and later reproduction; do not rerun the
fixed-cohort scoring command as a tuning loop.
