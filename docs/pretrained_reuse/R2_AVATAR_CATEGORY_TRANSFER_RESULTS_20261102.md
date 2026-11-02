# AVATAR held-category transfer results (2026-11-02)

## Decision

The frozen-feature AVATAR readout transferred above four-choice chance on this
TRAIN-only category holdout, but performance varied sharply by category and was
well below the project's 95–98% aspiration. This is narrow category-match
evidence, not general audio/video decision quality, a natural-media runtime
result, or an architecture ceiling. It does not justify adding a larger gate or
promoting this route.

The experiment fits only the 73,857-parameter readout. EfficientAT and
VideoPrism feature vectors were frozen and loaded from existing caches. It ran
on CPU; no GPU training, feature extraction, cloud use, official validation, or
sealed-audit access occurred.

## Frozen protocol and integrity

The protocol is specified in
[`R2_AVATAR_CATEGORY_TRANSFER_PROTOCOL_20261102.md`](R2_AVATAR_CATEGORY_TRANSFER_PROTOCOL_20261102.md).
Source commit: `0155fab17b6ba601cb1764d29183aa823188e966`.

The 80 TRAIN categories were deterministically divided into 16 held categories
and 64 seen categories using seed 17. Source groups associated with held labels
were quarantined before fitting. The 64 seen categories were divided into fit
and inner-validation source groups. Candidate options were rebuilt separately
within each split. The three partitions contain 7,464 fit choices / 1,404 query
groups, 1,874 inner-validation choices / 351 groups, and 2,490 held-category
choices / 513 groups. Source-group, asset, and query overlap are all zero.
The held-category evaluation was performed once after the validation-selected
checkpoint was fixed; it was not used for selection or retuning.

All source inputs are pinned by SHA-256:

| Input | SHA-256 |
| --- | --- |
| TRAIN model input | `0139f4f59a685c3f83890156b59c42d12aca9db54b22ee5c0a9c70b6ec7d60ef` |
| TRAIN sidecar | `90adbf04450e0854fdb379dbcceb6aaa73790b205c08f1cf6c2cb786b6710a57` |
| Audio cache | `5736e2c7e094a93aea81fa13f517799292e790197dda6f6282140e3882636a58` |
| Video cache | `6ab47dcc0f9c8fe691425fa45a6f1394c8520677829a419c51b93a45da068b6e` |

The exact command and effective settings are retained in the external run
directory:
`C:\CodexArtifacts\pretrained-reuse-r2\avatar-category-transfer-20261102-attempt-03`.
The original training command was:

```powershell
python C:\CodexWorktrees\r2-perception-control-freezer-20261031\scripts\run_avatar_category_transfer.py `
  --train-input C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261102-attempt-01\cross-source-option-choices-02\train-model-input.jsonl `
  --train-sidecar C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261102-attempt-01\cross-source-option-choices-02\train-analysis-sidecar.jsonl `
  --audio-cache C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261102-attempt-01\cross-source-option-choices-02\audio-features-subset.npz `
  --video-cache C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261102-attempt-01\cross-source-option-choices-02\video-features-subset.npz `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\avatar-category-transfer-20261102-attempt-03
```

The run started at `2026-11-02T01:42:22.134142Z` and finished at
`2026-11-02T01:42:26.691058Z`. Python 3.11.9, PyTorch 2.6.0+cu124, NumPy 2.4.6,
Windows, CPU, and two Torch threads were used. `CUDA_VISIBLE_DEVICES=-1` was set
and zero CUDA devices were verified before fitting.
The fit-only normalization reconstruction utility committed with this result
has SHA-256 `2a2f9ae89144d191f7b77b1093564f08f1ef44dbecad2eb892df17621aff1178`.

## Model, fit, and measured results

The readout scores each option with `[audio 384 + video 768] -> 64 -> GELU -> 1`.
It has 73,857 trainable parameters. AdamW used learning rate `0.001`, weight
decay `0.0001`, batch size 128, at most 50 epochs, patience 6, and minimum
validation-NLL improvement 0.001. Fit-only means and standard deviations were
used for normalization, with a `1e-5` standard-deviation floor. The selected
checkpoint was epoch 7 by macro-category inner-validation NLL `0.204964`; the
run stopped after 13 epochs. Fit time was 2.5168 seconds; held-set inference
took 0.0095 seconds. The best readout checkpoint is 297,472 bytes. This is only
the head size; it excludes the resident encoders and is not the complete package
size.

| Evaluation | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Inner validation, macro over 55 seen categories | 0.9123 | 0.2050 | 0.1179 | 0.0794 | 0.9281 |
| Held categories, sample weighted, N=2,490 | 0.6960 | 0.7295 | 0.4116 | 0.1063 | 0.8021 |
| Held categories, macro over 16 categories | 0.7133 | 0.7133 | 0.3926 | 0.1466 | 0.8112 |

Four-choice chance is 0.25. A seed-17, 5,000-replicate bootstrap over 513
audio-query source groups gives the held-set sample-weighted Accuracy 95% CI
`0.6702–0.7222`. The roughly 20-point macro Accuracy gap between seen-category
validation and held-category test is consistent with a substantial category
transfer challenge for this setup. It does not isolate whether that gap comes
from category shift, data support, or the feature/readout combination.

| Held category | N | Query groups | Accuracy | NLL | Brier | ECE-15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baby laughter | 180 | 39 | 0.9111 | 0.2516 | 0.1433 | 0.0531 |
| Bark | 188 | 40 | 0.7447 | 0.5613 | 0.3365 | 0.0982 |
| Cello | 83 | 18 | 0.9759 | 0.0506 | 0.0275 | 0.0287 |
| Computer keyboard | 175 | 36 | 0.7086 | 0.7223 | 0.4082 | 0.1024 |
| Electric guitar | 170 | 35 | 0.7294 | 0.5879 | 0.3628 | 0.0853 |
| Fire engine, fire truck (siren) | 187 | 22 | 0.9626 | 0.1186 | 0.0581 | 0.0343 |
| Guitar | 246 | 58 | 0.7114 | 0.5541 | 0.3599 | 0.1070 |
| Gunshot, gunfire | 207 | 50 | 0.5507 | 1.1485 | 0.6503 | 0.2402 |
| Neigh, whinny | 27 | 9 | 1.0000 | 0.1678 | 0.0718 | 0.1379 |
| Nose blowing | 38 | 12 | 0.5000 | 1.1304 | 0.6266 | 0.2768 |
| Steel guitar, slide guitar | 180 | 32 | 0.7222 | 0.6335 | 0.3862 | 0.1183 |
| Tapping (guitar technique) | 199 | 41 | 0.6030 | 0.9641 | 0.5424 | 0.1730 |
| Train | 311 | 84 | 0.6977 | 0.6428 | 0.3916 | 0.0936 |
| Train horn | 164 | 51 | 0.3963 | 1.4759 | 0.7736 | 0.2539 |
| Tuba | 23 | 4 | 0.9130 | 0.2895 | 0.1504 | 0.1137 |
| Whistling | 112 | 21 | 0.2857 | 2.1146 | 0.9928 | 0.4297 |

The source-group support is uneven; in particular, Tuba has only 23 queries
from four groups. Per-category scores therefore should not be treated as equally
precise.

## Artifact integrity and recovery record

The run's fixed report, config, split, predictions, and best/last weights remain
unchanged. Their SHA-256 values are:

| Artifact | SHA-256 |
| --- | --- |
| `report.json` | `67a50c2f6a53724b7575e00768bb3d75de9683dc0c8d568d0d80520462cf9d35` |
| `effective-config.json` | `b946b6204c78681737e23cddc7c82a350c9dfda582cdab99d6bdcca62301d602` |
| `split-manifest.json` | `4ce36f0c9a5f97706abde7c165e4800c48adf621da34f9e442ed2464f291ffba` |
| `held-category-predictions.jsonl` | `f14ecddf6ab795740276deff47a9474e167da530a0075fe634a51914e64cd1d9` |
| `joint/best-state.pt` | `817494b3ac22187dbeaa47c6bc14c2d2b947e3c23fc078355dc0439788787af0` |
| `joint/last-state.pt` | `1c2d5b4d39cc6fd76bee9330a04195414051f83f283635ff3307e0aed77afb5e` |

The first run attempt failed before training because setting an empty
`CUDA_VISIBLE_DEVICES` value did not hide CUDA on Windows. The second failed
before fitting because direct script invocation could not resolve the
repository package path. Neither attempt took an optimizer step or wrote a
checkpoint. Both attempt records are preserved in their separate attempt
directories. The successful run used `CUDA_VISIBLE_DEVICES=-1`, verified no
visible CUDA devices, and fixed direct execution's repository-root import path.

The original runner did not persist the fitted normalization arrays. To close
that reproducibility gap without rerunning the model or evaluation,
[`complete_avatar_category_transfer_artifacts.py`](../../scripts/complete_avatar_category_transfer_artifacts.py)
reconstructs the fit-only statistics from the hash-pinned TRAIN inputs and
frozen split manifest. It verifies every reconstructed split's IDs/groups and
the original fit-choice-ID hash. The statistics use only fit rows. However,
NumPy materializes each compressed NPZ `features` member before applying row
indices, so the cached TRAIN feature rows from fit, inner-validation, and
held-category groups may all have been transiently present in memory. These
caches contain only the pinned source TRAIN asset pool. No official-validation
or sealed-audit files, labels, or predictions were read, and no metric or model
evaluation was rerun. The original `artifact-completion.json` overstated this
cache-read detail; the append-only correction below is authoritative. The
supplement was added without changing `report.json` or any checkpoint:

| Supplement | SHA-256 |
| --- | --- |
| `normalization.npz` | `0447f511c0aff464e4bdb630fd3040520bbe43ed466e0d3c0c29f612429778f3` |
| `artifact-completion.json` | `2b3d867599b47ed7a30970ee730911b728ad9139a3b8b08f42455c4b07f7ff2c` |
| `artifact-completion-correction.json` | `65159ec6c2bbf148faa36f53cf4f0e2108ac5beb63210e52df6e156f8de01b45` |

Reconstruction confirmed 7,464 fit audio assets and 6,946 unique fit candidate
video assets. The statistics are a deterministic post-hoc reconstruction, not
the original run's separately logged byte-for-byte normalizer state; downstream
use should still validate reload parity before relying on them.

## Scope limits and next gate

Repository verification for this increment: `ruff check .` passed; the full
CPU suite passed with 2,723 tests and 11 skipped. The focused AVATAR probe and
artifact-completion tests passed again after the cache-read correction (10
tests).

- This is an AVATAR category-match task with cached features and opaque class
  choices. The readout does not process natural-language questions or options.
- It does not prove broad video/audio reasoning, temporal understanding,
  synchronized cross-modal complementarity, or general Omni performance.
- The 116,400,280-parameter route count includes 1,961,239 audio encoder,
  114,365,184 video encoder, and the 73,857-parameter readout. The AV package
  exceeds the project's provisional 200 MB package target; the 297 KB head is
  not the whole model.
- The category holdout uses only the source TRAIN partition and is not a sealed
  audit. No official validation or sealed-audit metrics were loaded for this
  experiment.
- The next product-quality work should return to the roadmap's admitted
  task/source gates. This result alone does not justify another AVATAR seed,
  larger gate, encoder adaptation, or product promotion.
