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

The CPU execution path was added to the EfficientAT extractor while preserving
the CUDA default and its reservation gate. It was verified with an isolated
CPU/GPU same-input feature comparison and unit tests that fail if the CPU path
queries `torch.cuda`. The full local suite now passes **2,442 tests, 11 skipped**;
Ruff, `git diff --check`, model/dataset manifest validation, dataset-catalog
validation, and the example manifest audit pass. EfficientAT CPU environment:
Python 3.11.9, PyTorch/torchaudio 2.6.0+cu124, NumPy 2.4.6. CUDA was not
initialized by the CPU extraction process.

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

## Audio extraction and one-time frozen-readout result

The unrelated Orbitune worker PID 16364 remained active throughout the
diagnostic (RTX 3080 Laptop: 15,821/16,384 MiB used, 355 MiB free in the last
snapshot). It was not stopped, changed, or accessed by the experiment. Rather
than start a CUDA context while it was active, the pinned EfficientAT extractor
was extended with an explicit `--device cpu` path. The existing default remains
CUDA and still requires the HotUGPU reservation environment variable. An
exact-input backend check against an existing CUDA feature produced cosine
similarity 0.99999994 and maximum absolute feature difference 4.62e-7. This is
one input only; it does not substitute for general cross-backend validation.

Three immutable CPU attempts were preserved. Attempt 01 stopped before audio
lookup because the media-root argument was one directory too deep. Attempt 02
extracted 32/200 windows before a progress logger attempted to aggregate null
CUDA memory values. Attempt 03 completed 200/200 windows in 9.002 seconds;
EfficientAT had 1,961,239 parameters and its state hash was unchanged. Its
manifest SHA is `4569b23116691dfb99c614be2b3b51ee9d7b6e9a087f43db3ca7e0a2916c5539`,
weights SHA is `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`,
source commit is `a425fdce92572e602a1d5634799bd9f1f2efa806`, and feature-result
SHA is `390cdda100245e922916fb8b31434043ccc4a61b9394c72cd80ab9fa8a35a832`.
Per-record media/window identities and all 200 feature hashes were rechecked.

The frozen evaluator was then run once on the fixed, label-balanced cohort.
The 200 records contain 100 visible and 100 not-visible examples; three
two-second windows were boundary-shifted only enough to fit the media. The
evaluation receipt is
`C:/CodexArtifacts/pretrained-reuse-r2/perception-test-av-v2-extension-eval-20261025-attempt-01/evaluation-result-01.json`,
SHA-256 `f36a413db72c68393aff2dd147052081f2fd89343ca98b0f03f0ad5611a13c04`.

| Frozen head / control | Accuracy | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|
| Audio | 0.570 | 1.3568 | 0.3390 | 0.2941 |
| Video | 0.580 | 1.3719 | 0.3307 | 0.2985 |
| Joint Audio+Video | 0.665 | 1.1568 | 0.2869 | 0.2648 |
| Audio zero input | 0.500 | 0.6931 | 0.2500 | 0.0008 |
| Video zero input | 0.500 | 0.7162 | 0.2613 | 0.1061 |
| Joint, audio zeroed | 0.555 | 1.2418 | 0.3334 | 0.3065 |
| Joint, video zeroed | 0.600 | 0.8875 | 0.2744 | 0.2216 |
| Joint, audio shuffled | 0.615 | 1.3907 | 0.3190 | 0.2845 |

The paired video bootstrap used 10,000 repetitions, seed 17, and percentile
intervals. Joint minus Audio Accuracy was +0.095 (95% interval +0.015 to
+0.175), with NLL delta -0.200 (interval -0.494 to +0.088) and Brier delta
-0.052 (interval -0.115 to +0.010). Joint minus Video Accuracy was +0.085
(+0.015 to +0.160), NLL delta -0.215 (-0.447 to +0.011), and Brier delta
-0.044 (-0.095 to +0.007). Thus the point Accuracy gain over either single
head is supported in this cohort; calibration gains remain uncertain.

Input ablations are mixed. In the joint head, restoring audio over a zero-audio
input improved Accuracy by 0.110 (interval +0.045 to +0.175) and Brier by
-0.047 (interval -0.092 to -0.003), while its NLL interval crossed zero. Adding
video over the audio-only joint-head control improved Accuracy by 0.065 but
its interval crossed zero; NLL worsened by 0.269 (interval +0.044 to +0.510).
Replacing audio with a deterministic derangement changed Accuracy by +0.050
for the aligned input, but the interval crossed zero; the corresponding NLL
and Brier intervals also crossed zero. This does not establish that each stream
is reliably necessary for this decision.

The deliberately balanced cohort has 50% positive prevalence, unlike the
source's roughly 91.3% visible prevalence (about 94.3% among boundary-eligible
examples). Therefore these NLL/Brier/ECE values are conditional on the balanced
probe and are not source-prevalence calibration estimates. Selection used
already inspected official-validation labels; this is a nonblind exploratory
transfer diagnostic, not a sealed audit or product-quality result. The official
test split remained unopened. No training, calibration, checkpoint selection,
or sealed-audit evaluation occurred. The old Audio/Video/Joint readout,
normalization, manifest, corpus, and model weights were not modified.

Fresh output root (large weights/media remain outside Git):
`C:/CodexArtifacts/pretrained-reuse-r2/perception-test-av-v2-extension-eval-20261025-attempt-01/`.
