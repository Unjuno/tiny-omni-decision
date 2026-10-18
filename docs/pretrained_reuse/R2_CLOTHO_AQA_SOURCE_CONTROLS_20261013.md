# Source-coherent inference controls, frozen before audio results

Joint gains must be compared against independently trained question-only and
audio-only arms. Also predefine inference diagnostics on selected joint weights:
source-deranged audio and standardized audio coordinates set tozero (the train
mean representation, not a recorded silent waveform). Use the joint selected
temperature unchanged; controls cannot select checkpoints, sampling or calibration.

Derangement groups all questions for an underlying source. Sort source IDs,
normalize uploader IDs, then apply the existing fixed-seed grouped cyclic
permutation to uploader categories. Each source maps to exactly one different
uploader's source; source assets are bijective. No labels/confidence enter mapping.
For all560 validation questions (140sources), the per-query source marginal is
also preserved because every source has4questions. Full mappingSHA:
`82b6a398c80dbdc8fc0519262e82863454b033afb0b624789d9b4dbda38e4efd`.
The comparison runner checks this hash before fitting and saves actual mapping.

Mapping was also computed for training/calibration as a structural check.
Calibration550questions over138sources has unequal counts, so source bijection
does not preserve its per-query source marginal; no calibration derangement
is used. Calibration and all learning inputs remain the original paired data.

For source-deranged audio, reported scores use original paired question targets.
They measure sensitivity to incorrect evidence, not ground-truth quality of the
replacement source. A score drop does not alone prove reasoning. Report these
diagnostics with real joint/unimodal metrics and grouped uncertainty, retaining
all questions and any unfavorable result.

## Actual checks

Three targeted tests passed: source-coherent bijection, no same uploader, input
order/label independence, conflicting owner rejection and impossible grouping.
Ruff passed. Executed fixed-budget implementation replay with zero audio,
using full frozen question features and a separate parity output. Real, deranged
and zero-mean paths all reproduced the original question-only560 probabilities
bitwise; selected best weightSHA remains identical. This replay is software
validation and not a measured audio/joint model. Diagnostic overhead .004951s CPU.

Artifacts under `C:/CodexArtifacts/pretrained-reuse-r2/`:
`clotho-aqa-source-control-policy-01/` contains split mapping/hashes;
`clotho-control-implementation-parity-01/` contains protocol, replay outputs,
all three probability arrays and parity result. Audio acquisition02 remains live;
no real full audio comparison has run yet. CI for0acadf6 succeeded on both checks.
