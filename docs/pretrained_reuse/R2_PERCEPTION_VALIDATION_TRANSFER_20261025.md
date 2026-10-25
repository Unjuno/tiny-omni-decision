# Perception Test frozen-readout validation transfer diagnostic

## Scope and limits

This is a one-time, fixed-readout transfer diagnostic on 200 previously
unscored official-validation videos, selected as 100 visible and 100 not-visible
events. The source-validation inventory and label-feasibility checks were
previously inspected. Labels were used to construct the balanced cohort, so this
is **not blind**, is not a sealed audit, and cannot qualify product quality or
close R2 M-2. The visibility task also has a known strong sound-class prior; an
input ablation measures dependence of this fitted readout, not proof that both
modalities are necessary for the decision. Selection forces a 50/50 class mix,
while recorded boundary-eligible validation prevalence is about 94.3% visible
(91.3% on the full official-validation source). NLL/Brier/ECE on this cohort are
conditional on that balanced sample and do not estimate source-prevalence
calibration. No prevalence correction will be fit from this cohort.

No training, calibration, checkpoint selection, test access, or sealed-audit
access is part of this diagnostic. The original frozen Audio, Video, and Joint
readout files are read-only.

## Frozen inputs and window handling

- Selection: `C:/CodexArtifacts/pretrained-reuse-r2/perception-test-av-v2-extension-preflight-20261021-01/selection.json`
  SHA-256 `10cdc277cf417acba7e50245d8e0d189a6f60ce467c8d87dade42a41cb05e6d9`.
- Media validation: adjacent `media_validation.json`, SHA-256
  `8c4b4e30988edf23c365f80175480d20d90f7533a8f97c8725665b46ec10d67d`.
- 200 unique video IDs, 100 examples per target, zero exact audio/video SHA
  duplicates, all files hash/decode checked. Official test remained unopened.
- Audio and video use the same two-second event-centered interval; video uses
  eight uniform endpoint-inclusive frame times. Three events are too close to a
  clip boundary for a centered interval. They are retained and the interval is
  shifted only enough to fit both decoded streams. IDs:
  `video_3445`, `video_4083`, `video_3512`. The correction count is explicit in
  the manifest. This minor preprocessing difference is a transfer limitation.
- Frozen readout result, heads, and train-only normalization remain hash-pinned
  by the evaluator. No checkpoint or source artifact was modified.

## Code and preflight

Pipeline commit: `5601703` (`Add fixed Perception validation transfer
diagnostic`). The evaluator's final media-identity and exact-frame checks were
added in `8912066` (`Verify media identity in frozen transfer evaluation`)
before any frozen-head scoring. The pipeline includes a validation-only manifest
builder, per-example receipts, zero-input and deterministic audio-shuffle
controls, and a paired video-level bootstrap.
Unit tests verify cohort integrity, boundary retention, and deterministic
paired intervals. The full suite passed **2,440 tests, 11 skipped**; Ruff and
`git diff --check` passed. The commit was pushed to the roadmap PR branch and
the continuation branch. Public GitHub Actions runs for code commits
`56017039639df111d53df61213c826e9b92a0b43` and
`8912066efa1c1cd94d43176559af7f4ba3015c77`, and report commit
`9fc7504cc5abc02773165787a119d51e1d5b5cd9`, completed successfully (two CI runs
per commit). The unauthenticated local GitHub CLI was bypassed with the public
Actions API.

The effective inference manifest is stored at
`C:/CodexArtifacts/pretrained-reuse-r2/perception-test-av-v2-extension-eval-20261025-attempt-01/inference_manifest.json`,
SHA-256 `4569b23116691dfb99c614be2b3b51ee9d7b6e9a087f43db3ca7e0a2916c5539`.
It contains all 200 examples and records the three boundary shifts.

## Video feature extraction completed

The pinned VideoPrism CPU extractor was smoke-checked on two fixed events, then
run once on all 200 examples. The checkpoint has 114,365,184 parameters, SHA-256
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`; source
commit `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`. All 200 outputs are finite
768-dimensional pooled vectors and all selected media hashes matched. Runtime
was 471.978 seconds on CPU. Feature receipt SHA-256:
`e49c67c8d27cadbb91a06329a5d7029be48f1fe544ac2b40c22e3439ce9145af`.

Two incomplete smoke attempts are preserved and are not results: attempt 01
used a malformed temporary manifest newline; a second attempt reached Python
but lacked the repository root in `PYTHONPATH`. The corrected two-example
smoke passed in `videoprism-smoke-02`; no quality metrics were computed.

## Audio extraction state

Audio inference has **not started**. At the latest read-only GPU snapshot,
2026-10-25 19:46:32 JST, the unrelated Orbitune process PID 16364 (parent
31916) remained active, using 15,821/16,384 MiB. It was not stopped or changed.
A CPU-only wait loop is checking for no compute PIDs, utilization below 5%, and
at least 14,000 MiB free for two consecutive samples. Only then does it invoke
the existing HotUGPU reservation wrapper (SHA-256
`59b038dcc61317263cb972b3098ecf99b6b3fa111c6bc2d0ffe752fc93c5a06a`) to start
the pinned EfficientAT extractor. It does not initialize CUDA while the
unrelated process is active. The wrapper must still obtain its reservation and
its child must pass the extractor's own gate. Until then, no audio feature
receipt, head metrics, input-ablation metrics, or bootstrap result exists.

Fresh output root (large weights/media remain outside Git):
`C:/CodexArtifacts/pretrained-reuse-r2/perception-test-av-v2-extension-eval-20261025-attempt-01/`.
Do not call this diagnostic complete until the gated audio extraction,
single frozen-head evaluation, hash validation, and final report are finished.
