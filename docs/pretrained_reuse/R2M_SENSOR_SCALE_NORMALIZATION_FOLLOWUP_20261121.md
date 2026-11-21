# R2-M sensor-scale normalization follow-up (2026-11-21)

## Question

The first same-trial audio/vibration Decision pilot consumed sensor values in their raw cached integer scale. The three vibration axes had large, unequal offsets and magnitudes, while frozen EfficientAT audio embeddings were order-one values. This follow-up changes only the sensor input transform to per-axis z-scores computed on train-split tokens. It tests whether the scale mismatch was a material optimization confound. It does not change the task, examples, split, model, seed, optimizer, schedule, budget, selection rule, or evaluation IDs.

## Data and integrity

The source is Zenodo record `10.5281/zenodo.7937591`, used under its CC BY 4.0 release terms with attribution ([record](https://zenodo.org/records/7937591); [data descriptor and acquisition methods](https://www.mdpi.com/2306-5729/9/12/144)). The published descriptor specifies 16 kHz mono audio and a three-axis LIS3DHH accelerometer, whose int16 readings span approximately ±2.5 g; the run did not apply a physical-unit conversion. The per-axis z-score is an empirical train-only transform over the cached resampled sensor tokens.

Both attempts use the same frozen feature cache and pair index:

- Feature-cache index SHA-256: `61758bebaf53ad810737d31e7955820bf7094e2166460c5ce595ea0c475c6498`
- Pair-index semantic SHA-256: `d1e1c74a1114fc0d071f71a50876b852e8d3fce8919fde9696368d4ebf365276`
- Pair-index file SHA-256: `33664b32205626fe1ec2102961e82163d2f55586f6c0567c9f1c80c0f4d3e7bf`
- Frozen validation: 761 pairs / 1,522 candidate rows, on the same IDs and labels in both runs. It includes only two validation participants (386 and 375 pairs), so these exposed development metrics do not estimate population generalization.
- The pair index is restricted to train/calibration/validation; the sealed audit was not loaded or scored.

The source paper describes time alignment using cross-correlation after receipt-time logging. The released trial schema records one trial timestamp and sensor sample rate but does not expose a per-trial alignment lag or a correlation-quality value. Equal duration and a publisher synchronization description therefore do not let this pipeline independently verify every trial's alignment. No synchronization claim stronger than the source documentation is made.

## Change and fixed run configuration

Only this input transform differs:

- Attempt 01: cached sensor counts passed unchanged.
- Attempt 02: per-axis z-score using all 128-token sequences from each unique training feature once; 5,449 train features / 697,472 train tokens.
- Train-only means: `[-10516.0449, -7948.5952, 34.5053]`
- Train-only standard deviations: `[28.5897, 35.9121, 22.6059]`
- Calibration, validation, and prediction use those fixed train statistics.

Everything else is held fixed: seed 17; 178,563 trainable parameters; width 96 / four heads; AdamW at 0.001, weight decay 0.01; 80 epochs; batch 16; cosine schedule; same pair index, cache, and checkpoint selector. Best checkpoints are selected on calibration NLL; validation is only reported after selection.

The optional implementation flag is `--sensor-normalization train-zscore`; its default remains `none`, preserving the old run behavior. The normalizer copies arrays, rejects any non-iterative split and derives statistics only from unique train cache IDs. Preflight used 16 train examples for a forward/backward smoke, zero optimizer updates, and scored neither calibration nor validation. The fixed-budget run then completed all 80 epochs.

Source commit: `765ceb21aeb20e58d4a5e9268c67b03c66daad33`. Attempt 02 effective-config SHA-256: `af491cdc4e49279445df425eb8889c684b34721162b02fb412700eabc6065f77`. The launch-time source tree was dirty because of this narrow implementation; exact per-file source hashes are retained in `effective-config.json` and are not represented as a clean committed snapshot.

## Results

| Attempt / input transform | Selected epoch | Validation candidate Accuracy | Candidate NLL | Candidate Brier | Candidate ECE-15 | Paired-choice Accuracy | Paired NLL | Paired Brier | Paired ECE-15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 / raw counts | 16 | 49.93% | 0.693149 | 0.500001 | 0.000744 | 45.34% | 0.693150 | 0.500003 | 0.046675 |
| 02 / train-axis z-score | 14 | 50.72% | 0.692931 | 0.499784 | 0.006557 | 48.09% | 0.692710 | 0.499564 | 0.022493 |
| Difference, 02 − 01 | — | +0.79 pp | −0.000217 | −0.000217 | +0.005813 | +2.76 pp | −0.000440 | −0.000439 | −0.024182 |

Binary chance references are 50% Accuracy, `ln(2)` = 0.693147 NLL, and 0.5 Brier. The paired-choice result is below chance after normalization; candidate-row Accuracy is only 0.72 pp above chance. The tiny NLL/Brier changes do not establish a useful signal. Calibration ECE moved in opposite directions by metric level and remains low-information near uniform predictions. Participant-level paired-choice Accuracy was 49.48% for participant 15 and 46.67% for participant 9 after normalization. No confidence interval is estimated from only two validation groups, and this validation set has been exposed for development.

Both runs completed 80 epochs / 54,480 optimizer updates with the same 5,448 pair questions per epoch. The raw-count run took 1,017.3 s; the normalized run took 1,262.1 s. PyTorch peak allocation was 103,123,968 bytes for each; the longer wall time is not evidence of a compute bottleneck caused by the normalization arithmetic. The normalized attempt selected calibration epoch 14 (calibration NLL 0.69310993) and ended at epoch 80; its train CE stayed near 0.693. No cloud compute or package upgrade was used.

## Interpretation and next step

**The sensor-scale hypothesis is not supported as a sufficient explanation.** Train-only axis z-scoring yields a small movement on this fixed, already-exposed validation set, but paired-choice accuracy remains below chance and the selected model is effectively uninformative. This follow-up does not establish that the task lacks physical cross-modal signal; the task may be poorly posed, the frozen temporal features may erase timing, or trial-level synchronization may not be independently verifiable from released metadata. It also does not justify wider architecture, more epochs, or product admission.

The natural next development decision is to stop escalating this same-trial matching readout and evaluate a label-grounded activity Decision with matched audio-only, vibration-only, and paired controls, using only the already allowed iterative partitions. The supplied-activity experiments already point to substantial participant variation and late overfit; therefore any follow-up must keep participant-disjoint reporting and a fixed-budget control, and should first verify whether activity cues in the three accelerometer axes add information beyond the audio embedding. Keep the sealed audit unopened. Retain both attempts as development evidence, not as model-selection evidence for the product Teacher.

## Reproduction artifacts and verification

Attempt 01 is preserved at:

`C:\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-gated-decision-pilot-v1-attempt-01`

Attempt 02, preflight, comparison, checkpoints, curves and per-example predictions are preserved at:

`C:\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-gated-decision-pilot-v1-attempt-02-train-axis-zscore`

The attempt-02 external `artifact-index.json` covers 14 files and has SHA-256 `c87974360cc7db9686be3ff59eee0b48442d69475af246ea804aa8ef0d4f9499`. The key metrics file has SHA-256 `ddd05fed964b277dcbcba6e878794a933d8adf1007c0278f34874850c0ca6c13`; its best checkpoint hash is `217043f371cebbff627630a81922ceee579e2700262664a2b71544e2440657fa`; final checkpoint hash is `75f18f295bc0451cab8d0a4d972ffdeaea01f39e1c3f5b1d9af63c03ffc47906`; training-history hash is `a7d3b904c3e3fc7ae1fef701597db17b3a1e32041e5a0064c3423b9fcdcafd73`.

Focused CPU tests for the normalizer and paired comparison pass. Ruff and the full project suite are checked separately in the implementation commit. No source media, feature tensors, model weights, histories or per-example predictions are added to Git. No sealed audit, quantization, recovery training, or product promotion was performed.
