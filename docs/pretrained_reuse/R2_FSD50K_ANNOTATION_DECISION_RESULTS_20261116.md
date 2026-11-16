# FSD50K Annotation-Presence Decision Results (2026-11-16)

## Scope

This is a task-specific audio experiment under R2, not an end-to-end or multimodal product result. The four binary questions predict whether the FSD50K annotation manifest marks a clip with `Animal`, `Human_voice`, `Vehicle`, or `Music`. A negative target means “not annotated with this tag,” not verified acoustic absence. The fixed questions use typed options (`no_annotation` / `annotated`); no text is supplied to the readout.

No official evaluation labels or media were opened or used. The experiment does not complete R2, establish the 95–98% product target, or demonstrate text, image, video, sensor, or fused-input capability.

## Integrity and frozen inputs

- Source branch: `codex/fsd50k-budget-resume-20261116`
- Source commit before the loader correction: `91df7252` (`fix: apply retry buffer to FSD50K resumed budget`)
- Fit code SHA-256, including the correction: `2861bd447afa4cf37697276703ce18a99f2cda5451dee8e9d18534c21b1e98c4`
- Dataset revision: Zenodo `10.5281/zenodo.4060432`
- Frozen preflight manifest SHA-256: `0bbe5db5e6167a4c041e349889920f8da0f1a0703ee00fa63ff12a59a0531b6d`
- Acquisition receipt SHA-256: `518bbd43a109fedd0746ac1acbb82a380fe2b8869f620044b59681c6187ce34a`
- Complete-media acquisition validation: 18,879/18,879 files, 18,863 distinct media hashes, zero exact media-hash overlap across splits; official eval labels/media both false.
- Feature encoder: EfficientAT revision `a425fdce92572e602a1d5634799bd9f1f2efa806`; checkpoint SHA-256 `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`.
- Feature matrix SHA-256: `0301519c86ad56546492280cda9035e2a2ab31f51cde49d69645878499694971`; feature-row SHA-256: `1354b7694f64f2d03a97f275bfff8a557b391b7679ffcb1a94ee5f713452e815`; sample-order SHA-256: `70ac66fd0fe6963b67497ac1b3a7f6d2e9b3b6f0f2a20344594877ad04fccc33`.
- Frozen uploader-disjoint counts: train 12,845 clips / 2,089 uploaders; calibration 2,053 / 290; validation 3,981 / 572. Each validation clip contributes four binary questions (15,924 decisions).

The decision loader initially expected `official_eval_media_acquired` in the preflight JSON, while the pinned preflight records `media_downloaded` and `official_eval_labels_read`; the independent complete-media verifier records `official_eval_media_acquired`. The loader was corrected to enforce label blindness in preflight and enforce both label/media blindness in the matching acquisition-verification artifact. A regression test now verifies that preflight fails closed if `official_eval_labels_read` becomes true. The actual preflight and independent verifier were both checked before fitting.

## Fit protocol and artifacts

- Run: `C:\CodexArtifacts\pretrained-reuse-r2-c-local\fsd50k-annotation-presence-v1-decision-20261116-attempt-01`
- Device: CPU only; no GPU training or cloud compute.
- Environment: Windows, Intel Core i7-12700H, Python 3.11.9, PyTorch 2.6.0+cu124, NumPy 2.4.6.
- Seed 17; 20 epochs; batch 64 clips; AdamW; learning rate 0.001; weight decay 0.0001.
- The pinned EfficientAT encoder was frozen. The typed Decision readout had 25,089 parameters; encoder plus readout totals 1,986,328 parameters.
- One scalar temperature was fitted using calibration only. Checkpoint selection used minimum validation macro NLL. All 20 epochs ran; epoch 12 was selected. The final checkpoint remains separate from the selected checkpoint.
- Fit wall time: 32.41 seconds. Validation time was not separately instrumented. Monetary compute cost: $0.
- Result status: `PASS`; selected-checkpoint reload verification: true.

The actual preflight schema correction was tested before fitting. Focused tests: 6 passed; Ruff passed. No model weights, media, or checkpoints are stored in Git.

### Main validation metrics at selected epoch 12

| Tag | Accuracy | NLL | Brier | ECE (15 bins) | Balanced accuracy | Positive recall | Positive support / 3,981 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Animal | 96.01% | 0.1198 | 0.0625 | 0.0100 | 87.08% | 76.73% | 275 |
| Human voice | 94.02% | 0.1524 | 0.0872 | 0.0071 | 86.13% | 74.64% | 623 |
| Music | 94.83% | 0.1340 | 0.0768 | 0.0118 | 87.92% | 78.32% | 558 |
| Vehicle | 96.43% | 0.1139 | 0.0592 | 0.0143 | 79.63% | 60.49% | 243 |
| **Macro** | **95.32%** | **0.1300** | **0.0714** | **0.0108** | **85.19%** | **72.54%** | **1,699** |

The 95.32% macro accuracy is not a balanced quality result: positives are sparse, and balanced accuracy is 85.19%. An always-negative baseline reaches 89.33% macro accuracy (per-label: Animal 93.09%, Human voice 84.35%, Music 85.98%, Vehicle 93.90%) while its balanced accuracy is 50%. Thus the model adds about 5.99 percentage points of macro accuracy over that baseline, but still misses 39.51% of annotated Vehicle positives. Accuracy alone would overstate usefulness. Calibration ECE is low, but calibration does not repair the recall gap.

### Uploader-cluster uncertainty (post-selection validation)

To quantify sampling uncertainty without rerunning or tuning the model, saved selected-checkpoint predictions were resampled by uploader group (`group_id`): 10,000 paired nonparametric cluster-bootstrap replicates, seed 17, percentile 95% intervals. The 572 uploaders, rather than individual clip questions, are the resampling units.

| Macro metric | Point estimate | 95% cluster-bootstrap interval |
|---|---:|---:|
| Accuracy | 95.32% | 94.46–96.18% |
| Balanced accuracy | 85.19% | 82.71–87.53% |
| NLL | 0.1300 | 0.1112–0.1493 |
| Brier | 0.0714 | 0.0594–0.0838 |
| ECE (15 bins) | 0.0108 | 0.0105–0.0193 |
| Positive recall | 72.54% | 67.63–77.17% |

Per-label 95% intervals are also saved in the bootstrap artifact. Accuracy / balanced-accuracy intervals are: Animal 94.87–97.24% / 83.03–90.95%; Human voice 90.43–96.79% / 82.43–90.20%; Music 92.11–96.95% / 83.09–91.98%; Vehicle 94.60–97.75% / 72.37–86.50%.

These intervals are descriptive only. The same validation set selected epoch 12, so the intervals are post-selection and are not independent confirmation or sealed-audit evidence.

### Learning curve

| Epoch | Mean train CE | Validation macro NLL | Validation macro accuracy | Validation macro balanced accuracy | Selected |
|---:|---:|---:|---:|---:|:---:|
| 1 | 0.43 | 0.29 | 88.8% | 50.0% | |
| 4 | 0.23 | 0.24 | 91.1% | 56.8% | |
| 8 | 0.13 | 0.15 | 94.8% | 77.1% | |
| 10 | 0.10 | 0.13 | 95.3% | 84.7% | |
| 12 | 0.08 | 0.1300 | 95.3% | 85.2% | Yes |
| 16 | 0.06 | 0.13 | 95.3% | 84.7% | |
| 20 | 0.05 | 0.1311 | 95.4% | 84.9% | |

Training CE continued to fall. Validation improvement slowed after roughly epoch 10; epoch 12 minimized the fixed selector, while the final epoch did not improve it. Since train CE is not the same calibrated metric as validation NLL, this curve is a qualitative overfit/stagnation signal, not a directly comparable train-validation gap.

## Reproduction hashes

- Effective config SHA-256: `cecd5a3410c85075586a921dcf09b09f77e51e50451c597d6df484904ca27198`
- Training history SHA-256: `6b8662b84892c36fb0385dd36877f30fb4587da7aceb1785f3d78466961b617e`
- Selected validation predictions SHA-256: `fb01dc17507d400654a77003f24c942c7257e224a07ab79ccbc59e938ab05e4a`
- Best checkpoint SHA-256: `f159ab05b34cff3ed39f000de97d9ed08ea3fc97f4a83f50130cd45291a16d2c`
- Final checkpoint SHA-256: `46a8573cf8cb556c85dd3aac69c7913e791bdacf2c9a6c507e8d193a4dcb0921`
- Result JSON SHA-256: `49e53467e7ac51b265564eb0b4ecc49fa6141ffd3f2295450c83f236832b4c6d`
- Bootstrap analysis code SHA-256: `0451cdb6a44417718c811db1cf2f72684fdfc949147b3eff1b8bec1412b76d3a`
- Bootstrap artifact: `C:\CodexArtifacts\pretrained-reuse-r2-c-local\fsd50k-annotation-presence-v1-decision-20261116-attempt-01\validation-group-bootstrap.json`; SHA-256 `bd7db0b9450086fa97d1a3786cfbc561ea82ca51b24c030a39538f21d9a39230`.

## Interpretation and next gate

This is evidence that a tiny typed readout can exploit frozen whole-clip audio representations for these four annotation-presence questions. It is not evidence of 95% balanced accuracy, verified event detection, compositional candidate reasoning, or general audio decision competence. The strongest immediate concern is minority-positive recall, especially Vehicle. Before broadening to another modality or fusion, compare against a prevalence baseline and evaluate on a separately justified task with labels that represent actual acoustic presence/absence. Keep official evaluation sealed until the task and selection policy are frozen.

The sealed/pretrained reuse R2 objective remains incomplete. Text, image, video, sensor, and cross-modal decision quality have not been demonstrated by this run.
