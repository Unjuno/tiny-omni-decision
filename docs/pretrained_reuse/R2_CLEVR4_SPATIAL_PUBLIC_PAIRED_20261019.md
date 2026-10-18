# DINOv2 spatial vs global fixed public validation — 2026-10-19

## Result and limits

On the same 19,988 already-public Clevr-4 validation images, the previously
selected epoch-7 spatial-cell readout scored 92.25% macro Accuracy, compared
with 91.37% for its matched global-pooling control. Spatial minus global was
+0.88 percentage points in macro Accuracy, −0.0159 NLL, −0.0088 Brier, and
−0.0009 ECE. A paired image bootstrap favored spatial for Accuracy, NLL, and
Brier. This is a one-time comparison on an already-exposed public validation
set, not blind evidence, an independent audit, or product-quality validation.
No metric was used for training, calibration, checkpoint selection, tuning, or
promotion. The selected checkpoints had already been chosen on internal
validation before this public evaluation.

## Matched comparison

Both readouts use the same frozen DINOv2-small and MiniLM option embeddings,
input IDs, candidate order, preprocessing, and selected epoch. The candidate
heads are the two arms of the existing matched seed-17 experiment: 180,736
trainable parameters each, selected at epoch 7 from eight completed epochs.
No weights were updated during this evaluation.

| Readout | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Global-pooling control | 91.37% | 0.2108 | 0.1173 | 0.0137 | 0.9264 |
| Spatial 4×4 cells | 92.25% | 0.1949 | 0.1086 | 0.0129 | 0.9332 |
| Spatial minus global | **+0.88 pp** | **−0.0159** | **−0.0088** | −0.0009 | +0.68 pp |

Each taxonomy has 19,988 decisions. The score change is not uniform:

| Taxonomy | Global Accuracy | Spatial Accuracy | Accuracy Δ | Global NLL | Spatial NLL | Global Brier | Spatial Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Color | 98.29% | 98.83% | +0.55 pp | 0.0508 | 0.0359 | 0.0266 | 0.0186 |
| Texture | 95.72% | 95.28% | −0.44 pp | 0.1364 | 0.1535 | 0.0634 | 0.0707 |
| Count | 71.59% | 74.98% | +3.40 pp | 0.6506 | 0.5865 | 0.3774 | 0.3437 |
| Shape | 99.89% | 99.92% | +0.03 pp | 0.0053 | 0.0037 | 0.0020 | 0.0014 |

The count gain is the largest; texture regresses. Count remains far below the
project's provisional per-task quality target. The spatial scorer applies a
shared transform to each of the 16 cell vectors and sums the results. It is
permutation invariant over cells, so it can use the distribution of local
features but **does not encode absolute cell positions**. The test suite now
asserts this behavior; this result must not be described as proof of spatial
layout reasoning.

### Paired uncertainty diagnostic

5,000 paired bootstrap replicates (seed 17, percentile intervals, NumPy linear
quantiles) resampled validation image IDs and kept all four taxonomy decisions
for each image together:

| Metric, spatial minus global | Difference | 95% descriptive interval |
|---|---:|---:|
| Macro Accuracy | +0.00883 | [+0.00715, +0.01054] |
| Macro NLL | −0.01588 | [−0.01910, −0.01268] |
| Macro Brier | −0.00878 | [−0.01036, −0.00713] |

These intervals describe paired variation over this public validation cohort.
They do not account for model selection on internal validation, repeated use
of the public split elsewhere in the project, or dataset shift. ECE was
computed with the existing 15-bin rule and is reported descriptively; no ECE
interval was computed.

## Integrity and one-time evaluation recovery

The HotUGPU wrapper observed another process using 15,990 MiB and waited for a
cooperative yield. It then reserved 3,200 MiB and started the evaluation child
with 13,015 MiB free. The evaluator processed all 19,988 images and wrote all
79,952 probability rows. A post-inference media-overlap validator then failed
because the runner wrote `bytes_sha256` without the validator's required
`media_sha256` field. This was a metadata-schema error after inference, not a
model, data, or numerical error.

To avoid repeating GPU inference on these already-observed public labels, a
separate CPU finalizer checked the full ordered ID/taxonomy set, option order,
targets, finite normalized probability vectors, media hashes against the
previous fixed run, and split disjointness. It reproduced all metrics and
paired bootstrap values from the saved prediction file. The original
`failure.json` and initial run config were retained; the final run config
records the CPU recovery. No model inference was repeated. The evaluator now
writes both expected media-hash field names for future runs.

The option-embedding hash was not persisted before the first run hit that
post-inference failure. The final config records this as unknown rather than
reconstructing or guessing it. Peak VRAM was also not persisted by the failed
runner path. The exact model, readout, archive and input manifest hashes are
recorded below. The complete Clevr-4 archive was not rehashed in this run; the
previously verified SHA-512 receipt was used.

## Execution and provenance

- Branch: `codex/r2-independent-joint-task`.
- Runner source commit: `cda783a5709d135defd98a15c4309f546830d3f4`.
- Evaluated readout training config SHA-256:
  `b4b8f0be2dc4da0b3e976a1fdc92ce0725a175cdb5d20a2f33a1bd767eec93be`.
- Evaluated readout result SHA-256:
  `a6acad86fba04fb8feebf9830c4e60f4062641fb848ec3bee10d8144de29be7d`.
- DINOv2-small revision:
  `facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056`;
  weights SHA-256 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`.
- MiniLM weights SHA-256:
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- Global readout SHA-256:
  `3c109e2c027ea20e8177d6f5daf30cd282a9d44e5e73356423bdc62046a41a8f`.
- Spatial readout SHA-256:
  `22fdd1e72e2ffac1efe31cc4e172358fb78431aa06aac23390604bd8e0f16134`.
- Frozen parameters: 39,446,400. Trainable readout parameters: 180,736.
  Total unique pre-quantization model inventory: 39,627,136 parameters.
- Dataset: Oxford Clevr-4 100k official public validation, 19,988 images / 79,952 decisions. The same split had already been evaluated once for a different fixed readout; this paired result is explicitly exposed validation.
- Validation order SHA-256:
  `d289b6d0910344a633c9625c869e20a28601d37bf8c63a0652e65d7b56bda8e7`.
- Label manifest SHA-256:
  `16bd65147d13ab11f2359882b3447a388485d139049d4c375037d80ce43dbb43`.
- Saved prediction SHA-256:
  `cb6d9d11424aba22c4c5ea5f8d0d98d956757f722639e0bf64da8fa7d2c4fe1a`.
- Final run config SHA-256:
  `0afceb0b2368afb9402c6243f1d5b237ef13081dd51dde5934fc5842fcbd7ee6`.
- Result JSON SHA-256:
  `515bbea7a98f06e0d5561f5e3c5df50836d2144f49975dbd25d9e42a2ed22cc2`.
- GPU: NVIDIA GeForce RTX 3080 Laptop, 16,384 MiB. HotUGPU held a 3,200 MiB
  reservation. Inference loop reported 225.3 seconds; timestamps span about
  226 seconds from run-config creation through prediction completion. CPU
  finalization took about 24 seconds. Python 3.11.9, PyTorch 2.6.0+cu124,
  CUDA 12.4, Transformers 5.6.2, NumPy 2.4.6, Pillow 10.4.0. Cloud cost: $0.
  Peak VRAM is UNKNOWN because it was not saved on the failed runner path.
- Artifact directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\dinov2-clevr4-spatial-public-paired-attempt-01`.

## Decision

The one-time paired public comparison supports keeping the spatial-cell readout
as a promising development candidate for this synthetic image-count task. It
does not prove general image reasoning, absolute spatial reasoning, a blind
quality improvement, or multimodal readiness. Texture regressed and Count is
still weak. Do not fit, tune, promote, or quantize against this validation
split. Next quality evidence should come from a newly admitted, independent,
asset-grouped task and split; multimodal joint-input quality remains open.
