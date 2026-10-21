# Fresh CLEVRER Reasoning v1 — preparation and execution status

**Status: BLOCKED before feature extraction/training.** This is an engineering
status record, not a model-quality result. No checkpoint was trained or
selected, and no holdout media or labels were loaded.

## Work completed

- Built a deterministic whole-scene train/development/holdout selection from
  pinned CLEVRER source revision
  `98b842082ba4f7c18b6b9e3f39145871782a65ef`, seed 17.
- Each partition contains 200 scenes and all three reasoning question types.
  Train has 2,632 binary choices / 799 parent questions; development has 2,792
  / 846; held-out has 2,843 / 851.
- Train/development/holdout scene identities and normalized question/candidate
  content are pairwise disjoint. The recovered validation exposure inventory
  still has a 36-scene discrepancy from its source-screen hash; 2,748 possible
  legacy validation aliases were conservatively excluded. This split is not
  described as a perfectly reconstructed exposure inventory.
- Downloaded and CRC/size/SHA-checked the 400 train/development videos (about
  504 MB). SHA overlap against 4,591 historical local media identities is 0;
  train/development media overlap is 0. The range-read archive was not fully
  hashed. Holdout was intentionally not fetched.
- Added a separate VideoPrism frozen-feature task config, split/fetch gates,
  candidate-readout evaluation path, and a one-time paired scene bootstrap
  evaluator. Holdout fetch and feature extraction are separate post-selection
  paths; standard training extraction does not load the holdout manifest.
- Added compatibility for the fresh media-fetch report schema and regression
  tests for split grouping, report gates, reproducibility, holdout isolation,
  and paired bootstrap behavior.

## Frozen inputs and hashes

- Source commit: `33afecd` (`codex/r2-clevrer-fresh-reasoning-20261021`)
- Effective config: `configs/pretrained_reuse/path_a_videoprism_clevrer_fresh_reasoning_v1.yaml`
  SHA-256 `78ad4bbf56469d6b18e8db51f22d72c9bab9daf1823da0d2c5893abf5b9993b4`
- Preflight directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\clevrer-fresh-reasoning-v1-attempt-01`
- Preflight report SHA-256:
  `7dc36a9b369f90a6fbaf6d653d8e3ef7d1dc629ab661904108badcb886ae40d8`
- Train rows SHA-256:
  `24ee284cbe572b6bcf4d00bf7f796fe0448a04692276a1dc0752008f99079368`
- Development rows SHA-256:
  `3d704685a95cb50665f524f316155f9909fe1783b1bdf39e6f3328c7d7dea448`
- Isolated holdout rows SHA-256:
  `c7b32c946efab2880131111fd48fcab1372a1b48fdc0ba729062c5142a7d28a9`
- Train/development media fetch report SHA-256:
  `38395070c8be83cf996e8ae6bb8562380ac7b82daa00221c9be70a8dcbe8e3e8`
- VideoPrism checkpoint SHA-256:
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`
- MiniLM weights SHA-256:
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`

## Execution blocker

The local device is an NVIDIA RTX 3080 Laptop GPU (16 GiB). The pinned
VideoPrism source environment is Windows Python 3.11 with `jax==0.6.2` and
`jaxlib==0.6.2`, but has no JAX CUDA plugin installed. HotGPU correctly waited
for the existing GPU owner, reserved 3.2 GiB, and launched the feature
extractor. The child then spent several minutes using CPU while GPU utilization
remained 0% and only HotGPU's reservation appeared on the device. To avoid
mistaking CPU execution for the requested local-GPU measurement and to release
the reservation, the feature extractor child was stopped. No other process was
stopped or modified.

No feature cache was produced. The trainer preflight therefore stopped at the
missing expected cache path, before model fitting. The frozen-feature cache,
validation-selected readout, and held-out comparison are all unrun. No
quality conclusion can be drawn from this attempt.

## Verification and next safe step

- Ruff: passed for the full repository.
- Full CPU suite: 2,330 passed, 11 skipped.
- CI-equivalent model/dataset manifest, catalog, and license audit commands: passed.
- GPU environment: blocked by missing CUDA-enabled JAX backend; no dependency
  upgrade was attempted.
- Training, checkpoint selection, holdout fetch, and holdout evaluation: not
  run.

Resume only after providing a compatible local GPU execution environment for
the pinned VideoPrism revision (or an already-supported GPU extractor that
produces the same pinned feature contract). Re-run the feature extractor under
HotGPU, validate the resulting cache, then perform the frozen-readout
preflight/training. Fetch and evaluate the isolated holdout only after the
development-selected checkpoint is frozen. Do not change the split, config,
selector, or model based on holdout observations.
