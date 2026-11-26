# FSD50K frozen-feature annotation Decision result — 2026-11-26

## Result and scope

A single seed-17 Path A run completed on frozen EfficientAT audio features. It fits only a 25,089-parameter typed-option readout and predicts whether FSD50K dev metadata assigned each of four supplied labels (Animal, Human_voice, Vehicle, Music). For each label the readout returns probabilities over `[no_annotation, annotated]`.

This is an annotation-convention task: a missing FSD50K dev tag is **not** verified acoustic absence. The result is not a claim of broad audio understanding, audio-video fusion, or general Omni Decision quality. FSD50K's official evaluation labels and media were not opened or acquired. This is one development validation, with no independent test and no multi-seed uncertainty estimate.

## Frozen data and feature provenance

- Dataset: FSD50K, Zenodo DOI `10.5281/zenodo.4060432`.
- Frozen manifest directory: `C:\CodexArtifacts\pretrained-reuse-r2\fsd50k-annotation-presence-v1-preflight-20261115-attempt-01`.
- Preflight manifest SHA-256: `0bbe5db5e6167a4c041e349889920f8da0f1a0703ee00fa63ff12a59a0531b6d`.
- Seed 17 uploader-group split: train 12,845 clips / 2,089 uploaders; calibration 2,053 / 290; validation 3,981 / 572. The selector never used official eval.
- Acquisition attempt 10 independently verified all 18,879 selected files against SHA-256 and ZIP CRC; 18,863 unique media hashes; exact media-hash overlap across train/calibration/validation was zero. The acquisition verification also records `official_eval_labels_read=false` and `official_eval_media_acquired=false`.
- Frozen encoder source revision: `a425fdce92572e602a1d5634799bd9f1f2efa806`; checkpoint SHA-256 `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`; 1,961,239 parameters; 384-dimensional output.
- Feature policy: whole clip, native audio resampled to 32 kHz, float32 features. Feature receipt binds the exact manifest, acquisition verification, sample order, rows, checkpoint revision and checkpoint hash. Result: 18,879 rows, 18,879 unique source IDs, 2,951 uploaders, 590 shards. Extraction wall time 890.99 s, decoder time 1,711.11 s, peak allocated VRAM 908,866,048 bytes. A prior one-clip CUDA smoke measured 1.16 s forward and 26,054,656 peak allocated VRAM bytes.
- Feature artifact directory: `C:\CodexArtifacts\pretrained-reuse-r2\fsd50k-annotation-presence-v1-features-20261126-attempt-10`.

Key feature hashes:

| Artifact | SHA-256 |
| --- | --- |
| feature matrix | `0301519c86ad56546492280cda9035e2a2ab31f51cde49d69645878499694971` |
| feature rows | `1354b7694f64f2d03a97f275bfff8a557b391b7679ffcb1a94ee5f713452e815` |
| feature receipt | `a5a8dbba7b49667b7f9ef55642bd75d912d5a089956fc7e65bf4a75b74b437a5` |
| sample order | `70ac66fd0fe6963b67497ac1b3a7f6d2e9b3b6f0f2a20344594877ad04fccc33` |
| acquisition verification | `eddee9fe318b048ff18887c8ee22e3bacfb2a7fdb6cbf96bacd05ad965f9fce8` |

## Training and selection

- Run directory: `C:\CodexArtifacts\pretrained-reuse-r2\fsd50k-annotation-presence-v1-training-20261126-attempt-12`.
- One frozen-feature scorer, hidden width 64; only 25,089 readout parameters trainable. Encoder + readout total: 1,986,328 parameters.
- Seed 17; 20 epochs; 64 clips per batch; 201 updates/epoch, 4,020 updates total; AdamW, learning rate `1e-3`, weight decay `1e-4`.
- One scalar temperature was fitted on calibration only. The predeclared selector chose the lowest validation macro NLL after that calibration; best epoch 15 (3,015 updates), not final epoch 20. The best checkpoint was reloaded and verified. All 20 epochs ran; there was no performance early stop.
- Measured fit time: 23.33 s. The training artifact records `device=cuda`, but does not capture a run-specific GPU/driver/software-version snapshot or peak training VRAM; those values are **UNKNOWN**. No compute cost was incurred.
- The exact historical CLI invocation was not serialized; treat it as **UNKNOWN**. The effective configuration and its hash are saved. The execution checkout was based on `4a9e56e5e2bd84f15d4379fb9c8b86ea7fc7e3e7`; the trainer source used by the run has SHA-256 `f79f51b65936a7672f78fa7d251b13ef89b7ffcffcd41f7d81b400b1dcb188b0`. The follow-up commit includes this exact source and its regression tests.

## Validation results

All values are macro-averaged equally over the four binary questions unless shown per label. NLL is two-option cross-entropy; Brier sums squared error over both option probabilities; ECE uses 15 confidence bins. The comparison baseline predicts `no_annotation` for all four questions, using the training-set positive prior for its probabilities.

| System / validation metric | Accuracy | Balanced Accuracy | NLL | Brier | ECE-15 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train-prior / all-negative baseline | 89.33% | 50.00% | 0.3421 | 0.1924 | 0.0449 |
| selected readout, epoch 15 | 95.43% | 84.37% | 0.1281 | 0.0696 | 0.0094 |
| final readout, epoch 20 | 95.16% | 84.69% | 0.1381 | 0.0753 | 0.0128 |

Epoch 15 was selected by NLL, not Accuracy. NLL improved from 0.2944 at epoch 1 to 0.1281 at epoch 15, then worsened to 0.1381 at epoch 20 while training CE continued to fall. This is a mild late overfit/calibration-loss signal, not a claim about broader capacity.

| Label | Validation count / positives | Accuracy | Balanced Accuracy | NLL | Brier | ECE-15 | Positive recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Animal | 3,981 / 275 | 96.61% | 85.22% | 0.1108 | 0.0553 | 0.0098 | 72.00% |
| Human voice | 3,981 / 623 | 93.97% | 84.53% | 0.1561 | 0.0887 | 0.0067 | 70.79% |
| Vehicle | 3,981 / 243 | 96.13% | 79.47% | 0.1186 | 0.0606 | 0.0112 | 60.49% |
| Music | 3,981 / 558 | 95.00% | 88.24% | 0.1270 | 0.0737 | 0.0098 | 78.85% |

The raw 95.43% macro Accuracy overstates rare-positive detection: macro positive recall is 70.53%, and the weakest positive recall is Vehicle at 60.49%. The selected train/validation macro Accuracy gap is about 2.83 percentage points; the balanced-Accuracy gap is about 8.69 points. These are in-sample versus validation diagnostics, not an independent generalization estimate.

Learning-curve checkpoints (validation values):

| Epoch | Updates | Accuracy | Balanced Accuracy | NLL | Brier | ECE-15 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 201 | 89.33% | 50.00% | 0.2944 | 0.1733 | 0.0430 |
| 6 | 1,206 | 92.73% | 65.35% | 0.1875 | 0.1038 | 0.0405 |
| 9 | 1,809 | 95.26% | 82.55% | 0.1376 | 0.0734 | 0.0134 |
| 12 | 2,412 | 95.50% | 85.90% | 0.1307 | 0.0701 | 0.0134 |
| 15 (selected) | 3,015 | 95.43% | 84.37% | 0.1281 | 0.0696 | 0.0094 |
| 20 (final) | 4,020 | 95.16% | 84.69% | 0.1381 | 0.0753 | 0.0128 |

The values are rounded for display; `training-history.jsonl` is the authoritative source for exact per-epoch values.

## Artifact hashes

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| best checkpoint | 106,486 | `70e594c51b096af86de64751bf74a6d0554a320ee5dccab727fcf085899e8d8e` |
| final checkpoint | 106,210 | `9be72e46683a296282e3f0053486d357a636da6f5991ab72325750d4b886fe12` |
| effective config | 1,475 | `b4dfe4909f0d5490b6e7c9efe9500e85be73e274d12ededf295372f2cf907c86` |
| training history | 142,423 | `498e95702e076be17cb34a25204d03344f3a546f94dcd761b6730d6c85d98f64` |
| selected predictions | 2,915,291 | `e39785e39ec5432f604e9016a42fc000d34546679e6ba05f8765b216ae6fdbbd` |
| result summary | 4,850 | `d304c8ed445314feae8262d9319c8cfc29d2ee57d0395d0783d2453844f15db3` |

## Verification and engineering corrections

- `py -3 -m pytest tests/test_fsd50k_annotation_decision.py -q`: 9 passed, including available CUDA device/metric checks.
- Ruff on the trainer and focused test: passed. `git diff --check`: passed.
- Attempt-10 exposed a CUDA device mismatch in the option-validity mask. Attempt-11 exposed CPU-resident ECE bins/accumulator during validation. The final run used fixes placing tensors on the feature/logit device. The preflight uses `media_downloaded`; eval-media isolation is checked in the independent acquisition verification. Regression tests cover the real schema and fail-closed eval gates.

## Interpretation

This is one narrow audio Path A readout diagnostic. It does not close R2's modality-native/fusion gate or validate the product's four-modality target. It does not justify a broad BERT or audio-only product sweep. Do not use this validation again to tune candidates. The official eval remains unopened. Preserve the result as research evidence; do not promote it as an Omni product model.
