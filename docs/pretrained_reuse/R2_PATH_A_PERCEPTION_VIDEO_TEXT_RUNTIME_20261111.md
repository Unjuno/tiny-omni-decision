# Path A Perception Test video+text runtime canary — 2026-11-11

## Scope and result

Implemented an isolated experimental video+text Decision route using the pinned
VideoMAE v2 Small encoder, the existing pinned MiniLM encoder, and the frozen
Perception Test readout. The real Path A JSONL dispatcher and worker passed a
three-request, official TRAIN-only runtime canary on the local RTX 3080 Laptop.
This establishes runtime wiring, typed probabilities, option permutation
equivariance, and same-video feature caching only. It does not establish task
quality, calibration, text-free video ability, or useful joint-modality
complementarity. It does not close R2-M or the direct text-free R2 modality
gates. No targets or labels were read; no training, validation, or sealed-audit
evaluation was run.

The existing Perception Test disposition remains controlling: its exposed
validation results must not be replayed or used for selection/tuning. This
canary intentionally used two records from TRAIN `video_10001` and did not
inspect their targets.

## Inputs and provenance

| Component | Provenance |
| --- | --- |
| VideoMAE v2 Small source | revision `706cc172d65ebd4dedbee3f9c0183a93df9fa125`; source file SHA-256 `d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb` |
| VideoMAE checkpoint | `vit_s_k710_dl_from_giant.pth`; SHA-256 `24fb71687fa3671b8387cadfbab0f72af695692e93cf1ecc82caa888626172` |
| Text MiniLM | revision `4ca70771034acceecb2e72475f72050fcdde4ddc`; weights SHA-256 `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433` |
| Path A assembly manifest | SHA-256 `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519` |
| Frozen readout | `D:/CodexArtifacts/pretrained-reuse-r2/perception-test-video-readout-attempt-01/best-checkpoint.pt`; SHA-256 `e9516be6d58ba82c3c4f111e59d6b724e856b3ac412bbc2c57b683d35deecb8c` |
| Readout identity | seed 17, selected epoch 2, 444,673 parameters; config SHA-256 `9d63d29fec6b56a5281ad7ec9d7ec6919b87b31a7e2fe430da7b00c2e73baa0a`; code SHA-256 `82d8462331dce8fc1e5cf9c8d042ba5f3413039ea9e8232e965cc9375f05de84` |
| Frozen training corpus | `D:/CodexArtifacts/pretrained-reuse-r2/perception-test-mcqa-patha-v0-attempt-01`; train JSONL SHA-256 `2f2fa8ed4e81baaadcede50cc4343f0f0c55a02dd63703d3313e8d2188cb60a7` |
| Canary media | TRAIN `video_10001.mp4`; SHA-256 `e1cd1b93731ae8c5f0b814217c724157fa7e7fc1accbecd52628e06dd1147ce6` |

The readout loader validates the checkpoint envelope identity (nested
`state_dict`, selected epoch, seed, and config hash) before strict loading. It
does not treat the envelope itself as a state dictionary.

## Runtime method

The worker decodes 16 endpoint-inclusive uniform frames through PyAV as RGB24,
resizes to 224×224, and normalizes by `/127.5 - 1`. A hook extracts the final
VideoMAE block grid `[8,14,14,384]`; the runtime uses the trained `fc_norm`
pooling path and flattens spatial-temporal tokens to `[1,1568,384]`. MiniLM
encodes `Question: {question}\nCandidate answer: {option text without A/B/C
prefix}` with max length 256 and masked-mean pooling. The frozen readout scores
exactly three supplied choices and returns a softmax distribution. The
VideoMAE feature grid is cached by video content SHA in a bounded LRU.

Task ID: `path-a-video-text-videomaev2-perception-test-experimental-v0`.
This is intentionally task-scoped and accepts exactly three options; it is not
a general-purpose video interface.

## Execution record

Source commit: `ca70e572b14e3a00fbd800b5f71d3f7df5d5b7e2`.

Successful command (all inputs and outputs remained outside Git):

```powershell
python scripts/smoke_path_a_perception_video_text.py `
  --corpus-dir D:\CodexArtifacts\pretrained-reuse-r2\perception-test-mcqa-patha-v0-attempt-01 `
  --media-dir D:\CodexArtifacts\pretrained-reuse-r2\perception-test-mcqa-patha-v0-attempt-02-media\train `
  --model-root C:\CodexArtifacts\pretrained-reuse-r1\videomaev2-small-audit-01 `
  --dependency-dir C:\CodexArtifacts\pretrained-reuse-r1\python-deps `
  --assembly-dir C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4-clean-copy-20261026-01 `
  --readout D:\CodexArtifacts\pretrained-reuse-r2\perception-test-video-readout-attempt-01\best-checkpoint.pt `
  --output-dir D:\CodexArtifacts\pretrained-reuse-r2\perception-test-video-text-runtime-20261111-attempt-06 `
  --device cuda
```

Four earlier unique attempts failed during integration and were retained without
overwriting: attempt 01 omitted worker arguments; attempt 02 passed a checkpoint
envelope instead of its nested state dict; attempt 03 supplied a 5-D token grid
to a 3-D readout; attempt 04 omitted the candidate-embedding batch axis. Each
failure has a `failure.json`; the implementation fixes have regression tests.
Attempts 05 and 06 passed. Attempt 06 is the resource-instrumented record.

| Measurement | Attempt 06 |
| --- | ---: |
| Requests | 3 (two questions for one TRAIN video, then a candidate permutation) |
| Wall time | 8.020 s (worker startup 1.687 s; first dispatch 7.397 s; warm dispatches 0.018 s and 0.017 s) |
| Peak CUDA allocation | 312,207,360 bytes (about 297.6 MiB) |
| Worker peak working set | 1,333,006,336 bytes (about 1.24 GiB) |
| Resident unique parameters | 39,987,783 (VideoMAE 22,153,286; MiniLM 17,389,824; readout 444,673) |
| Weight files | 115,687,687 bytes total (VideoMAE 44,334,609; MiniLM 69,569,488; readout 1,783,590) |
| Same-video cache | one miss, then two hits |
| Candidate permutation maximum probability delta | 0.0 |
| Runtime environment | Python 3.11.9; torch 2.6.0+cu124; CUDA 12.4; transformers 5.6.2; timm 1.0.15; PyAV 18.1.0; psutil 7.2.2 |

These are component weight-file totals, not a complete installable package size.
The three-request latency is a smoke measurement, not a stable benchmark.

## Output evidence

Attempt 06 output directory:
`D:/CodexArtifacts/pretrained-reuse-r2/perception-test-video-text-runtime-20261111-attempt-06`.

`result.json` records status `TRAIN_RUNTIME_CANARY_PASS`, start
`2026-11-11T02:37:08.030687Z`, finish `2026-11-11T02:37:16.145319Z`, request
SHA-256 `47a54ae10ea001c70e97c07ab48d0b414a88e81f6d5ae3bfef08d0c5760a6e61`,
response SHA-256
`3e3d0d7ab396e7e86b1114c63f5dc24198475dfc412ff3a16253e9275c34c6f2`, and
explicit `targets_or_labels_used: false`, `validation_or_sealed_audit_opened:
false`, and `training_performed: false`. Its source/code hashes bind the exact
runtime measured. The report and result file are kept as separate artifacts.

## Verification and limits

CPU focused tests passed: 31 tests. The full CPU suite passed: 2,907 passed,
11 skipped. Ruff passed for all changed Python files. GitHub Actions remains to
be checked after push.

No v1/v2 corpus, checkpoint, or artifact was changed. The existing readout and
assembly were read-only. No GPU training, validation scoring, sealed-audit
access, source download, or quality selection occurred. Because the historical
Perception Test validation has already been exposed and its task disposition
bars further validation-guided tuning, quality is deliberately left
unmeasured here. The next admissible step is a fresh, rights-cleared,
group-disjoint video task with a pre-registered train/validation protocol and
text-free controls; this runtime can then be reused only if that source/task
passes its gates.
