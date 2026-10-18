# UAV-DualCog source screen — 2026-10-19

## Disposition

**REVIEW ONLY; not admitted for training, validation, or product evaluation.**
No files were downloaded or opened. The public task release is explicitly
test-only and supplies no public train or validation task split. Using those
tasks to train a readout, choose a checkpoint, or tune a pipeline would consume
the benchmark test set. The additional scene assets do not come with released
training task labels that can replace that split.

## Evidence reviewed

- Official repository: `SmartDianLab/UAV-DualCog`, public code repository;
  branch `main` resolved to `2ce147af1b73dd64fc96e5cca25d3fddf1151b45`;
  its repository metadata labels the code package MIT licensed.
- Official Hugging Face dataset card: `Lozumi/UAV-DualCog`; platform metadata
  revision `b29cc248678386149c87100a94c961ccfb82d819` shows CC BY 4.0,
  12 released benchmark scenes, 512 landmarks, 4,096 image QA samples, and
  2,048 video QA samples.
- The dataset card states all currently released benchmark task files are
  test-only and there are no public train or validation task splits. It
  distinguishes the 12-scene task release from an 18-scene supporting scene
  asset pool; scene geometry/landmark assets are not labeled task splits.
- The dataset viewer reports a configuration error due to duplicated `test`
  split names. No viewer rows were inspected.

## Decision

Do not use this release to train or tune a Path A readout. The CC BY 4.0
platform label is not sufficient to admit the underlying media independently:
the official page points to simulated AirSim scene assets, and asset-level
rights and provenance were not audited. Reconsider only if the authors publish
a rights-clear, scene-disjoint training and validation task corpus, or if a
new independently authored task set can be built from authorized non-test
assets without reading the released task/test labels.

Primary references:

- Official code package: https://github.com/SmartDianLab/UAV-DualCog
- Official dataset card: https://huggingface.co/datasets/Lozumi/UAV-DualCog
- Official benchmark site: https://uav-dualcog.lozumi.com/
