# Native unique-coverage long comparison — results

## Result

The fresh-coverage arm completed all 32,768 optimizer updates and passed a
fresh-process replay audit. It was compared with the already completed repeat
control using the same 24,346 validation IDs, options, order and scoring path.
Both runs selected step 8,192 by the preregistered macro-NLL rule.

**The selected fresh-coverage checkpoint did not beat the repeat control.** Its
macro Accuracy was 36.38% versus 37.42%, macro NLL 1.6591 versus 1.6302, Brier
0.7215 versus 0.7057 and ECE 0.1322 versus 0.1261. The video scene bootstrap
also found higher Video NLL for fresh coverage at the selected checkpoint;
the accuracy difference was uncertain. The fresh arm's 32,768-step endpoint
had higher aggregate Accuracy and lower NLL than the repeat endpoint, but both
endpoints were badly overfit and the fixed selector rejected them. This does
not establish a clean coverage benefit or justify selecting the endpoint.

No sealed audit was loaded, no new seed was run, no budget extension was made,
and no Teacher was promoted. All result files, checkpoints and intermediate
outputs are under `C:\CodexArtifacts\pretrained-reuse-r2`; none are committed.

## Preregistered comparison

- Source commit: `2b052cf939485ba0c17d1848ad7e18563079ae1c`.
- Config: `configs/pretrained_reuse/native_unique_coverage_long_v1.json`,
  SHA-256 `82bde23c8dbb5bba070725763886e4a90f99214a00ec5a8432859f54a552a963`.
- Repeat control receipt: `native-long-budget-fit-01/result.json`, SHA-256
  `645bedbe73ed508fad68c4f636a41df6c8630cd996641c44ce2a87d31550250b`.
- Fresh-run initialization was the same original seed-17 readout and fresh
  AdamW optimizer. The repeat arm was not resumed or modified.
- Both arms used the same 32,768-update cosine schedule, eight one-example
  microbatches per update, 62 warmup updates, initialization, loss, option
  shuffling, frozen features, source/task/option-count occurrence quotas and
  validation IDs. The only experiment factor was the frozen TRAIN coverage
  pool.
- Candidate A added 6,144 unique TRAIN questions: 2,048 Text, 2,048 Image and
  2,048 Audio. It added no Video questions.
- At selected step 8,192 the fresh run had consumed 65,536 examples, covering
  all 12,800 unique rows. Across the full run it consumed 262,144 examples and
  repeated 249,344 presentations. The repeat arm covered 6,656 unique rows;
  the fresh arm covered 12,800. Unique modalities were respectively
  `2,048/2,048/2,048/512` and `4,096/4,096/4,096/512` for
  Text/Image/Audio/Video. CLEVRER held at 512 unique question rows grouped
  into 256 underlying video scenes in both arms.
- Unique counts by source in the fresh arm were CLEVR-4 4,096; CLEVRER 512;
  Lombard 2,048; STARSS 2,048; Open-Jev condition relevance 1,366; programmatic
  decisions 1,364; Typed Decisions curriculum 994; and original text 372.
  Per-task unique counts and exact full-horizon modality/source/task occurrence
  counts are in the comparison JSON.
- Shared full-horizon occurrence quotas were 65,536 each for Text, Image,
  Audio and Video; 32,768 each for Lombard and STARSS; 65,536 each for CLEVR-4
  and CLEVRER. This confirms that total draw ratios were held fixed while the
  unique questions changed.
- The pinned full train pool included 8,417 unique observations, 5,385 unique
  media assets and 5,461 modality groups in the fresh arm. The corresponding
  repeat-control totals were 5,107, 3,114 and 3,382.

## Learning curve

Metrics below are macro averages across modalities. The selected checkpoint
rule was macro NLL with `min_delta=1e-5` and earliest tie. Train-probe metrics
are descriptive over the fixed 12,800 unique TRAIN rows; they are not an
independent generalization-gap estimate.

| Updates | Examples consumed | Train-probe NLL | Validation Accuracy | Validation NLL | Brier | ECE | Best step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 2.0563 | 15.91% | 2.2227 | 0.7997 | 0.0812 | 0 |
| 512 | 4,096 | 1.5832 | 29.59% | 1.7171 | 0.7460 | 0.0762 | 512 |
| 1,024 | 8,192 | 1.5636 | 29.24% | 1.7147 | 0.7474 | 0.0887 | 1,024 |
| 2,048 | 16,384 | 1.5467 | 31.57% | 1.7376 | 0.7541 | 0.1156 | 1,024 |
| 4,096 | 32,768 | 1.4421 | 33.76% | 1.6709 | 0.7313 | 0.0980 | 4,096 |
| 8,192 | 65,536 | 1.2918 | 36.38% | **1.6591** | **0.7215** | 0.1322 | **8,192** |
| 12,288 | 98,304 | 1.0921 | 39.85% | 1.7830 | 0.7259 | 0.1612 | 8,192 |
| 16,384 | 131,072 | 0.9868 | 42.43% | 2.1362 | 0.7366 | 0.1863 | 8,192 |
| 24,576 | 196,608 | 0.7758 | 46.67% | 3.3639 | 0.7418 | 0.2378 | 8,192 |
| 32,768 | 262,144 | **0.6842** | **48.77%** | 4.1994 | 0.7310 | 0.2458 | 8,192 |

The trainer flagged train-probe NLL falling while validation macro NLL rose at
2,048, 12,288, 16,384, 24,576 and 32,768. From step 8,192 to 32,768, the
train-probe NLL fell from 1.2918 to 0.6842 while validation NLL rose from
1.6591 to 4.1994. Accuracy continued rising, but confidence quality degraded.
This is direct evidence of overfitting at later endpoints on the observed
development validation.

## Matched validation comparison

The table reports repeat-control → fresh-coverage values. The `selected`
comparison pairs step 8,192 in both arms. `fixed endpoint` pairs step 32,768.

| Selection | Modality | Accuracy | NLL | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: |
| Selected | Text | 21.60% → 21.91% | 2.0022 → 1.9667 | 0.8378 → 0.8262 | 0.0802 → 0.0799 |
| Selected | Image | 35.40% → 36.08% | 1.7989 → 1.7797 | 0.7353 → 0.7282 | 0.0766 → 0.0701 |
| Selected | Audio | 62.46% → 56.65% | 1.0706 → 1.1114 | 0.4655 → 0.5033 | 0.0769 → 0.0759 |
| Selected | Video | 33.75% → 34.75% | 1.5017 → 1.6174 | 0.7556 → 0.7945 | 0.1898 → 0.2448 |
| Fixed endpoint | Text | 24.19% → 25.21% | 3.2376 → 2.3127 | 0.9549 → 0.8783 | 0.2650 → 0.1908 |
| Fixed endpoint | Image | 54.69% → 60.55% | 1.9488 → 1.2536 | 0.6687 → 0.5289 | 0.2286 → 0.1319 |
| Fixed endpoint | Audio | 68.82% → 72.97% | 1.3634 → 0.9694 | 0.4402 → 0.3479 | 0.1576 → 0.0828 |
| Fixed endpoint | Video | 37.25% → 38.50% | 12.8189 → 12.1739 | 1.1739 → 1.1496 | 0.5727 → 0.5551 |

At the selected step, macro metrics changed by -1.04 percentage points
Accuracy, +0.0288 NLL, +0.0158 Brier and +0.0062 ECE. Fresh coverage reduced
Audio Accuracy by 5.80 points; Text/Image changes were small. Video NLL and
Brier worsened despite a 1 point Accuracy gain. This is not a better selected
checkpoint.

At the fixed endpoint, macro Accuracy rose 3.07 points and macro NLL fell by
0.6488; each modality's endpoint Accuracy increased. However, endpoint macro
NLL was still 4.1994 for fresh versus 4.8481 for repeat, and Video NLL remained
above 12 for both. These late endpoints are not suitable checkpoint choices.

## Video scene-cluster uncertainty

Paired bootstrap resampled 200 underlying validation scenes 5,000 times with
seed 17 and percentile 95% intervals. At the selected checkpoint, all-Video
Accuracy difference was +1.00 points (95% CI -4.25 to +6.50); Video NLL was
0.1157 worse (95% CI +0.0086 to +0.2294). For temporal-descriptive questions,
Accuracy was -0.50 points (95% CI -8.50 to +7.50) and NLL was 0.1985 worse
(95% CI +0.0417 to +0.3741). The Accuracy intervals include zero, and these
validation-based intervals do not account for checkpoint-selection bias or
training-seed variation.

At the final endpoint, all-Video Accuracy difference was +1.25 points (95% CI
-4.50 to +6.75) and NLL difference -0.645 (95% CI -2.547 to +1.291). The wide
intervals include zero and both endpoints are strongly miscalibrated.

## Execution, resource use and integrity

- Successful smoke attempt 03: 32 optimizer updates, 12,800-row train probe and
  save/reload output parity; 84.23 seconds total, 2.57 training seconds and
  61.98 evaluation seconds. Projection for the full run was 3,941.51 seconds,
  within the two-hour budget.
- Successful fit attempt 01: 32,768 updates, best step 8,192; start
  `2026-10-18T14:22:08Z`, finish `2026-10-18T15:06:36Z`; 2,667.81 seconds total
  (1,808.48 training, 812.24 evaluation, 12.13 data setup). Peak allocated
  VRAM was 209,224,192 bytes; peak reserved VRAM 329,252,864 bytes; process
  working set 3,750,207,488 bytes.
- Successful fresh-process replay 01: start `2026-10-18T15:06:59Z`, finish
  `2026-10-18T15:10:38Z`; 219.19 seconds total. Best/final validation, train
  probes, full TRAIN union and state-pair outputs reproduced exactly.
- Hardware/runtime: NVIDIA RTX 3080 Laptop GPU, 16 GiB, Windows, Python 3.11.9,
  PyTorch 2.6.0+cu124, CUDA 12.4, NumPy 2.4.6, psutil 7.2.2, float32 and
  TF32 disabled. The model had 196,254,874 pre-quantization parameters, of
  which 889,345 readout parameters were trainable. Cloud cost was $0.
- Orbitune's watchdog performs a cooperative yield after it detects a foreign
  GPU process; its trainer may appear briefly while the handoff completes.
  The first two smoke attempts were interrupted too early (attempt 01 after
  32 smoke updates; attempt 02 before update 1). Their outputs are retained
  and marked blocked. Attempt 03 waited for the handoff and passed. Orbitune
  processes/checkpoints were not stopped, signaled or edited by this work.
- Repeat-control receipt and all its listed artifacts still match their
  pinned SHA-256 digests. Teacher v1 and prior Teacher artifacts/configs were
  not written. No audit data, calibration, cloud compute or model quantization
  was used.

## Artifacts and interpretation

- Fit receipt SHA-256: `5c84121118239e7f2234167e0728c0ec6648550913920154d84649d89c26c7df`.
- Best checkpoint SHA-256: `782245c005af8f606823133120c0f24cd9ee1c14774076395a16d35bbc1d1515`.
- Final checkpoint SHA-256: `1bba702fa778e0225a7151b60e32816d05e3d6c77a6b678c4e573c5904b4ec74`.
- Replay receipt SHA-256: `9f4be1e939084dbaa8e52e3cec37598832b591ac7fc0a174af07b1ee777fd2a4`.
- Full source/task metrics, every checkpoint curve point, exact hashes, paired
  deltas and bootstrap values:
  `C:\CodexArtifacts\pretrained-reuse-r2\native-unique-coverage-long-comparison-04\comparison.json`
  (SHA-256 `5dfbd75fb7a125504d97eb26a4faa161232c12c8026522921e03986ee3051768`).
- The result is **mixed** for the preregistered coverage question. Added
  unique examples improved some fixed-endpoint metrics, but not the selected
  checkpoint; later training clearly overfit validation probabilities. It is
  one seed on an observed development set, not a clean generalization result.
  No next seed or 65,536-update extension is authorized by this finding alone.

GitHub Actions for source commit `2b052cf939485ba0c17d1848ad7e18563079ae1c`
passed: <https://github.com/Unjuno/tiny-omni-decision/actions/runs/39135228531/job/117698272286>.
