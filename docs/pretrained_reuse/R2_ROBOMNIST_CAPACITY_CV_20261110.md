# RoboMNIST AV-match readout-capacity CV — 2026-11-10

## Question and predeclared rule

The fixed rank-32 AV readout fit reached near-zero training CE but was highly
overconfident and weak on its already-observed development validation. This
experiment tested one capacity reduction without changing the frozen encoders,
features, examples, optimizer, update budget, task, or validation protocol.

- Source/task and the fixed episode-group split: the existing RoboMNIST AV-match
  preflight and manifest; SHA-256
  `3f4575ba41d944310e1de83cbdc68385390581723262245b97dc0d2c71db660f`.
- Input features: the existing complete frozen cache only. No raw media was
  opened or re-extracted.
- Fixed training settings: seed 17, AdamW, learning rate `0.01`, weight decay
  `0.0001`, gradient clip 5, full-batch updates, 400 updates per fit.
- Candidates: bilinear readout rank 8 and rank 32. The 32-rank run is the
  preserved reference; it was not overwritten.
- CV: five deterministic, condition-stratified folds of the 390 training
  `pair_unit_id`s (78 per fold). The 780 underlying train source groups occur
  in only one pair unit each. All four rows from a pair stay in one fold.
- Rank 8 is selected only if its train-only out-of-fold NLL improves by at
  least `0.02` and its Accuracy is no more than `0.01` lower than rank 32.
- Only after this rule selected rank 8 were the fixed 2,000-row development
  validation features loaded. The validation result did not select the rank,
  checkpoint, or calibration. The sealed audit was not opened.

## Execution and artifacts

- Branch: `codex/r2-robomnist-capacity-cv-20261110`.
- Base commit present when the run started: `2e3a7b1145d3477e7f2474a20d03da1adc8bf2fd`.
  The training script and fold module were uncommitted during that run; their
  exact SHA-256 values are captured in the external run's
  `run-implementation-provenance.json`. The exact training code files are
  committed unchanged with this report. `result.json` retains the base commit
  in its `source_commit` field; do not read that field as though it contained
  the then-uncommitted experiment files.
- Command:

  ```powershell
  python scripts/fit_robomnist_av_match_capacity_cv.py `
    --manifest C:\CodexArtifacts\pretrained-reuse-r2\robomnist-av-match-split-20261103-attempt-03\manifest.json `
    --cache C:\CodexArtifacts\pretrained-reuse-r2\robomnist-av-match-feature-cache-20261103-attempt-01 `
    --output D:\CodexArtifacts\pretrained-reuse-r2\robomnist-capacity-cv-20261110-attempt-01
  ```

- Output: `D:\CodexArtifacts\pretrained-reuse-r2\robomnist-capacity-cv-20261110-attempt-01`.
- Effective config SHA-256: `959169b802b067249629bbc3f8d78bbc7b65ab3c0b17e20010970aa20ea6d619`.
- Full result SHA-256: `314810f75deb3952926a672f7e39b5d7b0a690877cee597e17cd125fd78a232c`.
- Train-only OOF predictions SHA-256:
  `505572c0c9e5b054bae9c82ac8b4c07f156126565ece734d1453a6240f0f075c`.
- Selected rank-8 checkpoint SHA-256:
  `ea13b03bfbfe739f9eb64d1dc3a6383ee8a9f9fe84cec652dbdd8984cae1209a`.
- Candidate validation predictions SHA-256:
  `33d94eaa42af2a1d842873b62c7e30b85d47144fc682c5d3b45919633f924bba`.
- Training source SHA-256 at execution: fit script
  `1b471c39be46315ffba49f1f3095cf9eba7976b60a75fda0e93b9737d436132f`, fold
  module `fef5ae9fc8635bd3ce316ec9a22a7183969ce2b8b85c7001b1b7648a4d20878d`.
  Test module at execution:
  `bce5dd974c33297cde6e3a1b3423532eb3cc9a7cf5526acf0c00712d182af5d1`.
  The fit script and fold module in this commit match those execution hashes;
  the test file was subsequently extended with bootstrap regressions.
- Paired comparison file:
  `paired-comparison-reproducible.json` in the same output directory. It
  identity-matches all 2,000 rows against the preserved rank-32 prediction
  file and records a 2,000-resample seed-17 bootstrap over 500 pair units.
- Paired analysis script SHA-256:
  `8d7bfb7c2364782e6cea1e9b823e971f2ec5bb3395bcd5933e7906a277af1101`.
- Runtime: Python 3.11.9, PyTorch 2.6.0+cu124, NumPy 2.4.6; CPU only,
  `CUDA_VISIBLE_DEVICES` empty. RTX 3080 use was 0 MiB / 0%. No cloud compute
  or cost. The complete command took 40.5 seconds; OOF fits took 27.2 seconds
  and the full-train refit 2.73 seconds. Validation scoring time was not
  separately instrumented. No raw media or sealed-audit data was read.
- C: had about 0.76 GiB free before the run; experiment outputs were written to
  D:. The 2 GiB feature-cache reserve was not used or changed.

## Results

The OOF-only rule selected rank 8. OOF values are cross-validation over the
training partition, not a blind test.

| Train-only OOF model | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Rank 32 | 49.74% | 1.3015 | 0.7589 | 0.3397 | 0.8371 |
| Rank 8 | 48.78% | 1.2038 | 0.7562 | 0.3320 | 0.8178 |

Rank 8 met the predeclared OOF rule: NLL improved by `0.0977`, while Accuracy
fell by `0.96` percentage points, just within the `1`-point guardrail. It was
then refit on all 1,560 train rows for the fixed 400 updates. In-sample
training Accuracy was 100% and NLL was `0.0068`; this large in-sample fit is
not a generalization result.

| Same fixed development validation | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Preserved rank 32 | 53.35% | 1.2494 | 0.7217 | 0.3074 | 0.8335 |
| Rank 8 candidate | 49.75% | 1.1597 | 0.7336 | 0.3144 | 0.8119 |
| Audio-only preserved control | 50.00% | 0.6953 | 0.5022 | 0.0258 | 0.5258 |
| Video-only preserved control | 50.00% | 0.7004 | 0.5068 | 0.0442 | 0.5442 |

Candidate-minus-rank-32 paired bootstrap (2,000 resamples, seed 17, 500
four-row source-pair units): Accuracy `-3.60` points (95% percentile interval
`[-6.30, -1.00]`); NLL `-0.0898` (`[-0.1657, -0.0148]`); Brier `+0.0119`
(`[-0.0257, +0.0520]`); ECE-15 `+0.0070` (`[-0.0218, +0.0364]`). This is
paired descriptive uncertainty on previously observed development data, not
a fresh independent estimate.

On this set, rank 8's lower NLL comes with worse Accuracy and point Brier/ECE;
it is near chance and does not improve Accuracy over either unimodal control.
Rank 32 retains the small above-chance joint-Accuracy signal over the controls
recorded in the original report, while its calibration remains poor. Reducing
readout rank alone therefore does not resolve both the weak accuracy and
calibration problems. Do not promote either readout as product quality or
claim that parameter capacity is the sole bottleneck.

## Parameter and package scope

- Rank-8 trainable readout: 18,432 parameters; checkpoint 75,406 bytes.
- Rank-32 readout: 73,728 parameters.
- Frozen encoders remain VideoPrism-B 114,365,184 and EfficientAT DyMN-0.4
  1,961,239 parameters. The rank-8 AV path therefore has 116,344,855 resident
  parameters before runtime metadata or any additional modality. Its retained
  encoder checkpoint bytes remain about 465.6 MB; this experiment did not
  quantize, merge, or package the encoders and does not meet the provisional
  200 MB package target.

## Decision and next gate

Keep attempt 01, the original rank-32 artifacts, the frozen feature cache, and
all previous reports unchanged. Rank 8 is a measured capacity diagnostic, not
an adopted model. Do not run further rank sweeps or calibration on this same
validation set. The next R2 step must use a newly admitted task with clean,
independent source groups and appropriate synchronized-input controls; until
then R2 product-quality gates remain open. This result is limited to one
seed, one task construction, episode-disjoint groups, and development data
whose rooms/devices/scenes were not held out.
