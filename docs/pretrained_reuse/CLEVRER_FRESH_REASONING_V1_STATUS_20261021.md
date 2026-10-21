# Fresh CLEVRER Reasoning v1 — preparation and execution status

**Current status: development readout trained; one-time holdout media fetch is
in progress.** This is an engineering status record, not a general model-quality
claim. The initial snapshot below is historical; see the execution update at
the end for completed training and holdout handling.

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

The initial blocker was resolved by using the same pinned VideoPrism extractor
with its GPU explicitly hidden. The later execution details follow.

## Execution update — 2026-10-21

### Frozen feature cache and development selection

The initial HotGPU attempt confirmed that the pinned VideoPrism environment had
no CUDA backend. The same frozen extraction was therefore rerun with
`CUDA_VISIBLE_DEVICES=-1` and `JAX_PLATFORMS=cpu`; it completed on
`TFRT_CPU_0` without using the GPU. The 400 train/development videos produced a
768-dimensional cache in 573.25 seconds. Peak working set was 4,178,378,752
bytes. The external cache is
`C:\CodexArtifacts\pretrained-reuse-r2\clevrer-fresh-reasoning-v1-attempt-01-features-cpu\`.

- Feature cache SHA-256: `4f9116d41f0d68d6b0db043b7f8b68f782874d4f5fbe51fd1a3d17c7604b1f60`
- Cache report SHA-256: `893a7dddb2f435b82e15786d5c751440dde4f142bd03359a6e7ab6949c9cc167`
- Cache validation: 400 unique video assets; scene overlap 0; normalized
  question/candidate content overlap 0; train/development manifests match the
  pinned hashes above.

A new explicit local config
`configs/pretrained_reuse/path_a_videoprism_clevrer_fresh_reasoning_v1_cpu_local.yaml`
points at that cache and uses fresh `*-cpu-cache` preflight/run directories; it
does not alter the original config. Config SHA-256 is
`37147fca815f0c17f4433a80ef4f6db1ca49f53410fdad5300abc69605311693`.
The trainer preflight passed with 2,632 train choices, 2,792 development
choices, 200 scenes per split, 400 unique videos, and zero scene/media/content
overlap. No sealed audit was loaded.

The frozen MiniLM text encoder and VideoPrism encoder fed the existing 295,169
parameter shared candidate readout. Training was launched through
`scripts/run_with_hotgpu_guard.py`; it waited while external CUDA PID 26064 was
active, reserved 3,200 MiB after the process yielded, then launched the child
with about 13,015 MiB free. It completed 24 epochs / 1,008 optimizer updates:
63,168 presentations of 2,632 unique train choices (60,536 repeated
presentations). Best validation macro question-type NLL was selected at epoch
10. Readout training took 12.92 seconds; peak CUDA allocation was 154,231,296
bytes. Existing Teacher v0 and v2 artifacts were not modified.

Run output:
`C:\CodexArtifacts\pretrained-reuse-r2\clevrer-fresh-reasoning-v1-attempt-01-model-run-cpu-cache\`.
The run report and validation predictions are preserved there. On 2,792
development choices, macro-by-type Accuracy was 0.5153, NLL 0.6930, Brier
0.4999, and ECE 0.0131. Per type: explanatory 0.5269 / 0.6928 NLL, predictive
0.5194 / 0.6927 NLL, and counterfactual 0.4996 / 0.6935 NLL. The validation
scene-cluster bootstrap (10,000 replicates, seed 17) gives a 95% interval for
macro Accuracy of [0.4916, 0.5386]. This is near binary chance and does not
support a useful reasoning capability claim. It is a completed
development-selected checkpoint, not a product result.

### Holdout fetch and one-time comparison

After freezing the development-selected epoch-10 readout, the post-selection
holdout media fetch was started in a separate output directory. The first
attempt exposed a `--holdout-only` bug: the fetch loop unconditionally looked
up a `train` partition that is intentionally absent in holdout-only mode. It
stopped after writing one clip. A regression test was added, and the minimal
guard was fixed; the same output directory was resumed with its existing media
hash checked and reused. The first paired-evaluation invocation loaded the
isolated labels and computed both frozen models' predictions, but stopped
before writing results because the CLI called `main()` before defining its
bootstrap helper. No metric output or predictions from that failed invocation
were inspected or saved. The function order was corrected and covered by a
regression test. The identical comparison is being rerun only to complete its
report; checkpoint, preprocessing, holdout, and bootstrap conditions are
unchanged. This recovery attempt is recorded explicitly and will not be used
for tuning.
