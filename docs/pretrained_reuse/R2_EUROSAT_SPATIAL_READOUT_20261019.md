# EuroSAT spatial readout — 2026-10-19

## Purpose and scope

This follow-up tested whether a frozen, pretrained visual representation can
support a small fixed-label image decision task under a geographic split. It
does not test arbitrary text options, open-world reasoning, multimodal fusion,
or the product model. It is one image-only transfer result and does not qualify
the model for product use.

The first source screen rejected the RGB JPEGs because they did not contain
coordinates or acquisition IDs. A later audit found TorchGeo v0.8.0's
EuroSAT spatial split manifests. The split implementation documents a
longitude-based 60/20/20 partition; the pinned manifests list the RGB basenames
and were checked against all archive members. This correction supersedes the
initial split conclusion, without changing the historical record in the
[source screen](R2_EUROSAT_SPATIAL_SPLIT_SCREEN_20261019.md).

## Data integrity and split

- Dataset: official EuroSAT RGB archive, 27,000 JPEGs, ten fixed classes.
- Archive bytes: 94,658,721; SHA-256
  `b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90`.
- Split source: TorchGeo v0.8.0, source revision
  `1c11c73a87b40b0485d103231a97829991b8e22f`.
- Split manifest SHA-256: train
  `2db7d455afb8dcbca898ea19a00f1f90c091734efdbba89e22aaf24056da243f`,
  validation
  `6c758477604b7057a0fd990d7f6327b63b99a6725aac11a6a9d0174a7fdd8f0b`,
  test
  `de22dec83d350cac3b3e4ca8e285cb6733c81ab94bf5bcf9213a567993402452`.
- Counts: 16,200 train, 5,400 validation, 5,400 test. All 27,000 basenames
  partition exactly once; all images decode as 64x64 RGB; image-byte SHA-256
  is unique for every image, with zero cross-split byte-identity overlap.
- Test media bytes were hashed during split-integrity preflight before fitting,
  but no test features or predictions were produced then. EuroSAT's public
  class labels are encoded in its class folders and filenames. This is therefore
  a public, non-blind spatial test, not a sealed audit. Test metrics are reported
  once here and were not used to choose the checkpoint.
- Class counts are spatially imbalanced. In particular, Pasture has 115 test
  examples versus 118 validation examples. Use macro metrics alongside overall
  Accuracy.

| Class | Train | Validation | Public test |
|---|---:|---:|---:|
| AnnualCrop | 1,788 | 394 | 818 |
| Forest | 1,309 | 614 | 1,077 |
| HerbaceousVegetation | 1,854 | 330 | 816 |
| Highway | 1,698 | 462 | 340 |
| Industrial | 1,659 | 488 | 353 |
| Pasture | 1,767 | 118 | 115 |
| PermanentCrop | 1,716 | 477 | 307 |
| Residential | 1,927 | 566 | 507 |
| River | 1,507 | 662 | 331 |
| SeaLake | 975 | 1,289 | 736 |

## Fixed model and training

- Frozen encoder: `facebook/dinov2-small`, revision
  `ed25f3a31f01632728cabb09d1542f84ab7b0056`, weight SHA-256
  `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`.
  The exact pinned official model card declares Apache-2.0. This is distinct
  from the dataset imagery's EuroSAT/Copernicus source terms.
- Representation: 768-dimensional concatenation of the CLS token and mean of
  256 patch tokens; resize to 224x224 using the pinned processor.
- Readout: one linear classifier over the fixed ten EuroSAT class IDs; 7,690
  trainable parameters. Encoder frozen parameters: 22,056,576. Total
  pre-quantization parameter count: 22,064,266. No quantization was performed.
- Seed 17; 50 epochs; batch size 256; AdamW, learning rate 0.001, weight decay
  0.0001. Per-feature normalization is fit on train features only. Select the
  earliest minimum validation NLL. The selected checkpoint is epoch 17; epoch
  50 is saved separately.
- Optimizer updates: 3,200. Train example encounters: 810,000; unique train
  images: 16,200. The selected train Accuracy/macro Accuracy was 98.70%/98.74%.
  The final train Accuracy/macro Accuracy rose to 99.72%/99.73% while validation
  NLL worsened from 0.1241 at epoch 17 to 0.1398 at epoch 50. This is evidence
  of late overfitting; the selected epoch was not the final epoch.
- Validation at the selected checkpoint: Accuracy 95.59%, macro Accuracy
  94.44%, NLL 0.1241, Brier 0.0654, ECE 0.0051, mean confidence 95.49%.
- GPU fit loop including epoch validation took 5.46 seconds as recorded. Full
  process wall time and training peak VRAM are unknown because the initial
  process stopped at a strict GPU-to-CPU replay tolerance check before writing
  runtime telemetry. Artifacts were retained, replay tolerance was separately
  verified at 2e-6, and the run was finalized post hoc without retraining.

## One-time spatial test result

The selected epoch-17 checkpoint was evaluated once on all 5,400 public spatial
test examples using the same model preprocessing. No updates or calibration
were performed. The first launch was mistakenly made without HotUGPU; it loaded
the encoder weights to CUDA but stopped before opening any image because of the
reservation check. The next launch used a wrong model path and stopped before
creating output or opening images. The successful run used the existing
read-only R1 model directory, verified its weight hash, and was HotUGPU-gated.

| Metric | Validation, selected epoch 17 | Public spatial test |
|---|---:|---:|
| Accuracy | 95.59% | 93.83% |
| Macro / balanced Accuracy | 94.44% | 90.93% |
| NLL | 0.1241 | 0.1879 |
| Brier | 0.0654 | 0.0942 |
| ECE | 0.0051 | 0.0159 |
| Mean confidence | 95.49% | 95.31% |

Test class Accuracy: AnnualCrop 87.41% (n=818), Forest 98.98% (n=1,077),
HerbaceousVegetation 91.30% (n=816), Highway 93.53% (n=340), Industrial
98.87% (n=353), Pasture 61.74% (n=115), PermanentCrop 87.62% (n=307),
Residential 97.04% (n=507), River 93.05% (n=331), SeaLake 99.73% (n=736).
Pasture is a clear weak class on this split; overall Accuracy alone obscures
that weakness. The train-frequency prior achieved 9.39% test Accuracy and 10%
macro Accuracy; this is only a basic reference, not a competitive image model.

These results are below the project's aspirational 95–98% per-task objective
when measured by macro Accuracy, and they apply only to this fixed-label
EuroSAT task. They are not evidence that an under-200M model can meet the
general multimodal product target. The encoder plus readout has 22.06M
pre-quantization parameters, but package size after quantization was not
measured. Do not promote this artifact or generalize the result to text, audio,
video, sensor, arbitrary option wording, or joint-input decisions.

## Reproducibility and artifacts

Source commit: `383cf64107fc251c9353e33dc501894e22179db6`.
The evaluation code now has a fail-closed HotUGPU environment check before
model loading or output creation, with CPU regression tests. That guard was
repaired after the first failed launch; the successful evaluation used the
equivalent inline check. The result records source commit `383cf64107fc251c9353e33dc501894e22179db6`,
but the exact dirty working-tree evaluator hash at process launch was not
persisted and cannot now be reconstructed. Feature extraction and test inference ran through
`scripts/run_with_hotgpu_guard.py`; test inference reserved 3,200 MiB on an
RTX 3080 Laptop 16 GB. It opened 5,400 test images, processed batches of 32,
and took 71.01 seconds with a reported peak allocated VRAM of 330,529,280 bytes.
During the run, GPU telemetry briefly showed 100% utilization and 16,145 MiB
used; the additional process(es) were not attributed. The evaluation completed
without an error, but this concurrent load is recorded as an environment caveat.
No cloud compute was used; cost was $0. The run environment was Python 3.11,
PyTorch 2.6.0+cu124, Transformers 5.6.2, NumPy 2.4.6, and Pillow 10.4.0.

External artifacts (not committed):

- Split preflight: `C:\CodexArtifacts\pretrained-reuse-r2\eurosat-spatial-preflight-attempt-01\preflight.json`.
  SHA-256 `57c179243bd650448ae782bfa179da5e0cb27920b3dc1cbe97afa544e155912a`.
- Train/validation features and records:
  `C:\CodexArtifacts\pretrained-reuse-r2\dinov2-eurosat-spatial-features-attempt-01`.
- Fit history and separate best/final checkpoints:
  `C:\CodexArtifacts\pretrained-reuse-r2\dinov2-eurosat-spatial-readout-attempt-01`.
- One-time test outputs:
  `C:\CodexArtifacts\pretrained-reuse-r2\dinov2-eurosat-spatial-test-attempt-02`.
- Test features SHA-256 `cb29d9777b594a178d35ff30a5d7756689691715903d5efe47fb709a89de6e80`;
  prediction rows SHA-256
  `f10b18eccb5383b5d56c55f77bb733b96bf6b74991e02503b659eff7f29571d9`;
  test media manifest SHA-256
  `8a29013ae75cf7b9a05bbf98353feccf8f382c8fa841d5c2350a8c388cba076c`.
- Selected readout SHA-256
  `1f439b3c50c076ae97ebb225b4c9bd3beb71706859234a7f3b3f0cde0c45d88d`;
  training history SHA-256
  `84b859debb6752a85c5f42c7202624b907da299c3352fabdc3715dc8c99a4cb4`.
- Training config SHA-256
  `bf8f798cdcffa7221b3f5591e09a61ad23f8f3225fe9c8c329375818341978a5`;
  training result SHA-256
  `1dbe132c6b02e180c9b520519a582ee6c8edbfac42a1e6fee898dabb82668a9d`.
- Test config SHA-256
  `19ca3d2a314877e4e06a8cb022ea7aef34c55ab1813dca5aa7bac594ead53ef2`;
  test result SHA-256
  `a256cd7552df987c51aaff054272c505bb84a73dd330a8f275a46c52d6170d8e`.

Operational failures were preserved rather than hidden: an initial source screen
missed the TorchGeo geographic manifests; a training invocation was rejected
by an archive-hash typo before the successful run; the first training process
hit an overly strict replay assertion after completing all epochs and was
finalized from saved artifacts without retraining; and an initial test command
used the wrong model path and stopped before decoding test images. The final
test attempt passed the HotUGPU reservation and completed. A prior attempt
without HotUGPU did not decode or evaluate any image. The existing DINOv2 model,
feature, readout and source archives remain read-only; only new code and this
report were added to the worktree.

## Source references

- [EuroSAT official author repository and license](https://github.com/phelber/EuroSAT)
- [DINOv2-small exact pinned official model card](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056)
- [EuroSAT v2 archive record](https://zenodo.org/records/7711810)
- [TorchGeo v0.8.0 EuroSAT geographic split implementation](https://github.com/torchgeo/torchgeo/blob/v0.8.0/torchgeo/datasets/eurosat.py)
- [Pinned TorchGeo split-file revision](https://huggingface.co/datasets/torchgeo/eurosat/commit/1c11c73a87b40b0485d103231a97829991b8e22f)
- [Copernicus Sentinel data terms](https://cds.climate.copernicus.eu/licences/ec-sentinel)
