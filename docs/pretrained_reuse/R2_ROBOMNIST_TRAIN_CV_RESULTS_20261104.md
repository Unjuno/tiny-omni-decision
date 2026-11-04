# RoboMNIST TRAIN-only learning curves — 2026-11-04

## Measured result

Five fresh rank-32 bilinear fits completed all 400 updates on CPU. The pooled
out-of-fold curve shows increasing overconfidence: held-fold NLL rises from
0.7161 at step 50 to 1.2446 at 400 while training NLL falls. Accuracy rises only
from 50.19% to 53.21%; ECE-15 rises from 0.0833 to 0.3010. Every scheduled
held-fold NLL is worse than the uniform binary predictor (ln2 = 0.6931).
These are development diagnostics, not independent product scores. The findings
support investigating regularization/readout behavior before extending this
fixed model's training budget. They do not prove an encoder/model-family limit.

| Updates | Train Accuracy | Train NLL | Held-fold Accuracy | Held-fold NLL | Brier | ECE-15 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5210 | 1.6015 | 0.4994 | 1.6648 | 0.8445 | 0.3927 |
| 25 | 0.5949 | 0.6568 | 0.5045 | 0.7272 | 0.5303 | 0.1011 |
| 50 | 0.7260 | 0.6007 | 0.5019 | 0.7161 | 0.5207 | 0.0833 |
| 100 | 0.8788 | 0.3899 | 0.5173 | 0.8121 | 0.5763 | 0.1629 |
| 200 | 1.0000 | 0.0054 | 0.5295 | 1.2070 | 0.7071 | 0.2987 |
| 400 | 1.0000 | 0.0016 | 0.5321 | 1.2446 | 0.7135 | 0.3010 |

Train metrics are arithmetic fold means (all folds have equal size). Held-fold
metrics use concatenated out-of-fold predictions. Each of the five fits used
624 original TRAIN source groups / 312 intact pair units / 1,248 rows; each held
fold had 156 source groups / 78 pair units / 312 rows. Total coverage is 780
TRAIN groups / 390 pairs / 1,560 rows. All normalization was fit inside each
fold; each model was freshly initialized. Models trained on only 80% of original
TRAIN, so this is not an exact replication of the old all-TRAIN fit. The folds
share fitting data, and the entire source has prior project development exposure.
No checkpoint or budget was selected for final testing or product promotion.

## Corrected development gate

The earlier blanket conclusion that exposed original validation prevents any
further training was too strong. The original objective and EVAL_PROTOCOL permit
TRAIN-internal development. This experiment preserves the original validation's
closure, all historical runs, and the sealed audit. New independent data remains
necessary for a fresh product-quality claim. The same-repetition task is an
AV correspondence diagnostic, not semantic robot action understanding, scene-OOD
performance, or a general Omni benchmark. The full R0–R8 goal remains open.

## Reproducibility and resources

- Preregistered [protocol](R2_ROBOMNIST_TRAIN_CV_PROTOCOL_20261104.md).
- Source commit: `4b4276982202f4b8444f60c5a718c599ada4b429`; implementation hashes are in `run.json`.
- Original manifest SHA-256: `3f4575ba41d944310e1de83cbdc68385390581723262245b97dc0d2c71db660f`.
- Effective config SHA-256: `cb57ae69b53a06a1f0b43c6373ef1126b22c89afa8956f300936bdf5808e13cb`.
- Prediction JSONL SHA-256: `9851c3080929241650c3dd601c09ace2003be6453fe8c3bc1353ef2e5a234bbf`.
- Start: `2026-11-04T10:18:23.366324+00:00`; finish: `2026-11-04T10:18:48.951292+00:00`.
- Runtime: 25.60 seconds for five fits, scheduled metrics and
  checkpoint reloads; feature loading/preflight and Python startup excluded.
- Three update timing samples: [0.020105900010094047, 0.013655399961862713, 0.012650199991185218]; mean
  0.015470 seconds/update. These are short CPU samples, not a
  steady-state GPU throughput claim.
- CPU: `Intel64 Family 6 Model 154 Stepping 3, GenuineIntel`; Windows-10-10.0.26300-SP0; Python 3.11.9, Torch 2.6.0+cu124,
  NumPy 2.4.6; FP32, four Torch threads, deterministic algorithms.
- Readout: 73,728 trainable parameters;
  296,584 bytes/checkpoint. All 30 scheduled checkpoints
  were reloaded with exact held-fold probability equality.
- Peak process working set observed by the 500-ms Windows monitor:
  610,983,936 bytes
  (582.7 MiB). No GPU or backbone
  loading; local/cloud monetary cost $0.
- Frozen encoder provenance retained in config: VideoPrism-B 114,365,184 +
  EfficientAT 1,961,239 parameters; with readout, the AV path totals 116,400,151.
  This is component accounting, not a full four-modality package measurement.
- New artifact directory: `C:\CodexArtifacts\pretrained-reuse-r2\robomnist-train-cv-20261104-attempt-01`. Contains effective config/fold map, history,
  out-of-fold predictions, fold normalizers, and all checkpoints. A sibling
  `.monitor.json` records command/PID/exit and observed memory. Original cache,
  checkpoints and validation arrays were unchanged. No sealed audit was opened.

```powershell
python scripts/fit_robomnist_train_cv.py --manifest C:/CodexArtifacts/pretrained-reuse-r2/robomnist-av-match-split-20261103-attempt-03/manifest.json --cache C:/CodexArtifacts/pretrained-reuse-r2/robomnist-av-match-feature-cache-20261103-attempt-01 --output C:\CodexArtifacts\pretrained-reuse-r2\robomnist-train-cv-20261104-attempt-01
```

Machine-readable small results: [JSON](R2_ROBOMNIST_TRAIN_CV_RESULTS_20261104.json)
and [CSV learning curve](R2_ROBOMNIST_TRAIN_CV_RESULTS_20261104.csv).

## Verification

Eleven focused CPU tests passed, including train-only whitelist enforcement,
fail-before-read overlap rejection, deterministic pair-group folds, fold-only
normalization and existing readout training/reload checks. Ruff passed. Full
suite/CI results are recorded after completion below.

## One-factor score-scale follow-up

After the baseline, a separate preregistered arm changed only the cosine
multiplier from sqrt(32) to1. Config values (other than scale), all fold IDs,
all fold-normalizer hashes, and parameter counts match. Both arms used fresh
seed17 initialization. This changes optimization and confidence together; it is
not post-hoc calibration. Unit-scale confidence is bounded at about88%.

| Updates | Unit-scale Accuracy | NLL | Brier | ECE-15 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4917 | 0.7341 | 0.5379 | 0.1307 |
| 25 | 0.5160 | 0.7057 | 0.5114 | 0.0712 |
| 50 | 0.4981 | 0.7174 | 0.5226 | 0.1003 |
| 100 | 0.4917 | 0.7320 | 0.5361 | 0.1115 |
| 200 | 0.5032 | 0.7355 | 0.5388 | 0.1136 |
| 400 | 0.4942 | 0.7354 | 0.5386 | 0.1175 |

At the fixed400 endpoint, NLL improves1.2446→0.7354 and ECE0.3010→0.1175,
while Accuracy falls53.21%→49.42%. NLL still exceeds the uniform0.6931.
Do not adopt this arm as a quality improvement. It reduces overconfidence but
provides no useful discrimination improvement on this development diagnostic.
The mean/std representation still discards sequence order. A separate next
hypothesis can test ordered versus shuffled per-stream recurrence, with matched
capacity, identical TRAIN folds and fit-only normalization. No LSTM has been
trained, and no product/final evaluation is authorized by these diagnostics.

- Source: `e749b0317a9e23d08162bfc6be37b93fecea91cd`; config SHA-256 `6326ee2d4cf01efbee01282311c81839c24a6b691d28f877e72c973c66e01e2d`.
- Prediction SHA-256 `337606bc374740e7203e0d206a34291c8a41956d9dcde53fe2a040466f8e766b`.
- Output: `C:\CodexArtifacts\pretrained-reuse-r2\robomnist-train-cv-unit-scale-20261104-attempt-01`; status COMPLETE; all30 reloads exact.
- Runtime 26.33s; observed peak working set
  610,930,688bytes; GPU/cloud cost0.
- Exact command: same baseline command above, replacing output with this unit
  directory and adding `--score-scale 1`.
- [Small result JSON](R2_ROBOMNIST_TRAIN_CV_UNIT_SCALE_20261104.json).

Baseline implementation: full CPU suite **2,873 passed /11 skipped**, Ruff passed,
and all28 workflow manifest/catalog validation commands passed. The scale-arm
addition passed its focused8 tests including multiplier-only parity; CI covers
the final committed tree. Total measured fit/report runtime across both arms is
51.93s, with10 fits/4,000 optimizer updates
and60 retained scheduled checkpoints. Neither original validation nor audit was
evaluated. Both runs remain development artifacts; the overall goal is active.
