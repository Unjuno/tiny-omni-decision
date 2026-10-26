# STARSS23 spatial Decision readout results (2026-10-26)

## Status and scope

The pinned, room-disjoint STARSS23 development corpus completed all three
frozen-feature readout arms on the local RTX 3080 Laptop GPU. Execution passed;
the task-quality result is weak and does not support a multimodal product claim.
The run used only the STARSS23 dev-train subset, its train/calibration/validation
split, and frozen audio/video features. Official dev-test, evaluation, and
sealed-audit data and labels were not read. Validation was used for candidate
reporting and is a development set, not a blind audit.

This task asks whether one event class occurred in one of eight spatial sectors
over a five-second synchronized audio/video window. It is a binary supplied-
option Decision task, not general audio/video understanding. The positive rate
is low, so the `always no` baseline has 97.3037% validation accuracy; raw
accuracy alone is not useful evidence.

## Frozen inputs and provenance

| Input | Scope / hash |
|---|---|
| Source | STARSS23 v1.1, DOI `10.5281/zenodo.7880637`, MIT release; pinned dev-train only |
| Window manifest | 2,720 five-second windows; result SHA-256 `8f08d3467bc49677fa432733b1bd3217b90056b649ad5c898202b6c97497c1a0` |
| Option corpus | 96 `["no", "yes"]` event-sector queries per window; result SHA-256 `7df56dd1dc472b8668b6b7c5cd7eb684e405771f88f4b8029656610c61d5a` |
| Audio encoder | DyMN checkpoint SHA-256 `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`; source commit `a425fdce92572e602a1d5634799bd9f1f2efa806` |
| Audio features | FOA DyMN, `[2720, 4, 384]`; SHA-256 `0d6aee00ded4db333f0e38b1d7cbee28cc70171683a4f487563cd102a17f9729` |
| Video encoder | VideoPrism checkpoint SHA-256 `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`; source commit `c6eba0ece3dd49a24ffc44582966997a4a59f1d8` |
| Video features | VideoPrism spatial tokens, `[2720, 8, 768]`; SHA-256 `257a11a297aa79f2c509b2ca7cda67e4583545da454dc88fdfa6ab4930c46748` |
| Video extraction record | PASS, 77 recording shards; 4,332.11 CPU seconds; no annotation labels read |
| Source commit for fit | `71beb89bd97f46c813322b038a8636a508f0343b` |

The split has 187,680 train queries from 49 recordings / 6 rooms / 1,955
windows; 26,784 calibration queries from 8 recordings / 1 room / 279 windows;
and 46,656 validation queries from 21 recordings / 2 rooms / 486 windows.
Recording, room, and audio/video asset identities are disjoint across splits.
There are only two validation rooms, so room-generalization estimates remain
uncertain.

The fit used seed 17; 128 hidden units; batch size 1,024; at most 40 epochs;
patience 6; AdamW learning rate 0.001 and weight decay 0.0001; equal positive
and negative CE in training; and train-window-only normalization. The selected
epoch minimizes calibration macro event-sector NLL with `min_delta=1e-5`.
Temperature and yes-intercept are selected on calibration only. All three
selected checkpoints were saved and reloaded before validation scoring.

The invocation and effective configuration hashes are in the run record:

- Run output: `C:/CodexArtifacts/pretrained-reuse-r2/starss23-spatial-readout-v1-attempt-01`
- `run_inputs.json`: SHA-256 `5cd3b99c2de930beb6b3cb5d448d76fe5ee0be9b5129b37dc52938b1448ff4a5`
- `effective_config.json`: SHA-256 `512bf60040ff7c8d75e95450fbcb04ea181fd6f780d1515d719603805cc21121`
- `result.json`: SHA-256 `694026a2747f3fc849f8a831eefad8a1ff622d3f3295ac92c9b386e6793b0d85`

The child training invocation ran under `scripts.run_with_hotgpu_guard` after
the competing GPU owner cooperatively yielded. Its exact argument list is the
`command` field in `run_inputs.json`; the requested device was `cuda`.

Runtime was Python 3.11.9, PyTorch 2.6.0+cu124, CUDA runtime 12.4, Windows 10,
RTX 3080 Laptop GPU with 16 GiB. The three fit times were 27.60, 47.96, and
53.43 seconds; evaluation/save times were 4.75, 6.54, and 9.01 seconds. Peak
allocated / reserved VRAM was 118.9 / 146.0 MiB for audio, 277.5 / 370.0 MiB
for video, and 329.6 / 554.0 MiB for joint. No cloud compute or cloud cost was
used.

## Development validation results

“Raw” means the selected head's direct probabilities. “Calibrated” uses the
temperature/intercept selected on the separate calibration split. Balanced
accuracy, sensitivity, NLL, Brier, and ECE accompany accuracy because of the
2.6963% validation positive rate.

| Arm | Trainable params | Best / stop epoch | Validation raw Acc / balanced Acc | Raw NLL / Brier / ECE | Validation calibrated Acc / balanced Acc | Calibrated NLL / Brier / ECE |
|---|---:|---:|---:|---:|---:|---:|
| Audio only | 199,681 | 13 / 19 | 76.62% / 75.51% | 0.5054 / 0.3264 / 0.0817 | 97.29% / 50.11% | 0.1063 / 0.0514 / 0.0079 |
| Video only | 789,505 | 8 / 14 | 13.27% / 49.79% | 0.7345 / 0.5413 / 0.3940 | 97.30% / 50.00% | 0.1244 / 0.0525 / 0.0009 |
| Audio + video | 986,113 | 7 / 13 | 18.56% / 53.24% | 0.7244 / 0.5312 / 0.3551 | 97.30% / 50.00% | 0.1230 / 0.0524 / 0.0021 |

The selected checkpoint SHA-256 values are audio `efaad09dfed4511791b5e0b253b1b7ef259998e15c2efadc544c80cfb9150040`, video `f4ebde85d4e39cc081d22f8894bef5adcd8ad68228fa437ae6c444f19f941f82`, and joint `f38375ea046f1265f4a0e9f35acbc0ec928a0c42c2678cceb0f00d8ca944c9c3`. Each arm's final checkpoint and saved predictions have separate hashes in `result.json`.

The calibrated arms predict `no` at the 0.5 decision threshold for every
validation query (joint: zero sensitivity). Their high accuracy is the class
prior, not successful event localization. Calibration improves probability
NLL/Brier over the `always no` baseline in this split, but it does not make the
system a useful yes/no decision rule at that threshold. Full event, sector,
room, and event-sector metrics plus per-example probabilities are preserved in
`result.json` and each arm's `validation_predictions.npz`.

The frozen encoders account for 1,961,239 DyMN and 114,365,184 VideoPrism
parameters. The joint path therefore evaluates about 117.31M parameters when
counting both frozen encoders and the 986,113-parameter head. These are
unquantized source parameter counts, not a packaged or quantized product size.

## Corrected input-control audit

Inspection after the fit found that the original zero-modality controls mixed
separately normalized audio/video vectors into a joint model trained on a
jointly normalized vector. Those two zero controls were out-of-distribution and
are not valid ablations. The original attempt and its files remain unchanged.

The control construction is corrected in code and tested. A no-training audit
reused the exact joint best checkpoint, exact train-fitted joint normalization,
fixed validation IDs, and seed-17 same-room video permutation. It zeroed a
modality by setting its coordinates to zero in the joint standardized space
(the train mean). Raw and already-selected calibrated probabilities are both
reported:

| Joint input | Raw Acc / balanced Acc | Raw NLL / Brier / ECE | Calibrated Acc / balanced Acc | Calibrated sensitivity |
|---|---:|---:|---:|---:|
| Intact audio + video | 18.56% / 53.24% | 0.7244 / 0.5312 / 0.3551 | 97.30% / 50.00% | 0.00% |
| Video set to train mean | 39.56% / 57.08% | 0.7513 / 0.5571 / 0.2022 | 97.30% / 50.00% | 0.00% |
| Audio set to train mean | 14.92% / 50.99% | 0.7244 / 0.5311 / 0.3748 | 97.30% / 50.00% | 0.00% |
| Video reassigned from another recording in same room | 18.24% / 50.18% | 0.7257 / 0.5325 / 0.3570 | 97.30% / 50.00% | 0.00% |

This fixed validation diagnostic does not show useful video grounding: removing
video improves raw thresholded accuracy, while changing which same-room video
is paired has little effect. Removing audio also leaves poor performance.
Because this is a single fitted head and exposed development validation, it is
not a general claim about the encoders or architecture. The corrected audit is
at `C:/CodexArtifacts/pretrained-reuse-r2/starss23-spatial-controls-audit-01`;
its `result.json` SHA-256 is
`50e0a8bc3aaa109a977e482dc50d6be68ea22b71da9ca61fbdd990e39402c2f5`, and its
prediction archive SHA-256 is
`3aea06fefbb4f281c188b3ad81ad8ef81bcd1f5472127dcd4d7998e0f0f5ec3b`.

The training script now builds zero controls in the joint normalization space
and stores both raw and calibrated control metrics/probabilities for future
runs. Tests cover the modality boundary and control result serialization.

## Interpretation and remaining gates

- This execution passed, but the candidate did not demonstrate useful video or
  audio/video joint Decision quality on this task.
- Audio-only is the strongest arm by a wide margin on raw validation, but its
  75.51% balanced accuracy is far below the project's 95–98% aspiration and is
  not cross-task evidence.
- Overall accuracy and calibrated ECE alone are misleading for this 2.7%
  positive-rate task; always-no behavior must remain an explicit baseline.
- The design has only two validation rooms and one calibration room, and the
  corpus has severe positive imbalance. A new split or repeated tuning on these
  validation IDs is not justified.
- No shared-Lens merge, temporal architecture, model quantization, or product
  promotion is supported by these results. Investigate label/task alignment,
  feature-to-azimuth evidence, and a naturally recorded task with adequate
  independent positive support before another AV fit.
- This one task does not satisfy text, image, audio, video, or sensor modality
  quality gates and does not close R2 or the overall roadmap.

The original features, corpus, checkpoints, and attempt-01 results are retained
read-only. The corrected control audit is a separate output, and no sealed or
official test data was accessed.

Repository verification after the control correction: Ruff passed; the full CPU
suite passed (2,612 passed, 11 skipped); training/evaluation dataset catalog,
example model manifest, and example dataset manifest validators passed.
