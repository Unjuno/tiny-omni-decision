# Fresh PhysicalAI action decision preflight — 2026-11-03

## Purpose and disposition

Prepare one new, bounded frozen-feature Decision evaluation of the existing
video+robot-state action-choice task on previously unused episode groups. The
task, feature backbone, 100 ms target horizon, four windows per episode, and
eight train-only action prototypes follow the already frozen
[`R2_PHYSICALAI_STACK_PLATFORMS_PILOT_PROTOCOL_20261102.md`](R2_PHYSICALAI_STACK_PLATFORMS_PILOT_PROTOCOL_20261102.md).
This is a fresh data-budget check, not a product admission or a claim that
video+state is required.

The source remains `REVIEW_NOT_ADMITTED`: source metadata has no scene/layout
identity, and a prior label-blind neighbor review found a shared simulated
workspace across sampled episodes. Results can only describe episode-held,
same-source development performance. The newly reserved cohort will remain
unopened. No data acquisition, feature extraction, fitting, or inference has
started at this preflight checkpoint.

## Source and frozen split

- Dataset: NVIDIA `nvidia/PhysicalAI-Robotics-Manipulation-SingleArm`,
  `panda-stack-platforms`, revision
  `73c00c18e7ca108e91846294b20a1d7cd5de9340`.
- Pinned publisher rights in the source screen: CC BY 4.0; attribution is
  required.
- Parent split manifest:
  `manifests/pretrained_reuse/physicalai-singlearm-stack-platforms-split-v3-20261102.json`,
  SHA-256 `3a2849afaa3cefa414c9dbfc094976e0f1cc48ad0e9636d65573fa306d1ed967`.
- New manifest:
  `manifests/pretrained_reuse/physicalai-stack-platforms-fresh-action-split-20261103.json`,
  SHA-256 `fb51eb0028c8f55372930548d2e480bbf616debc1a59779f29098d86ce1bbe7f`.
- The deterministic selector uses seed 17 and protocol ID
  `physicalai-stack-platforms-fresh-action-v1`. It selects 4,000 train, 1,000
  validation, and 1,000 reserved-development episodes from the original
  source-train partition. The original validation (2,645 groups) and original
  protected reserve (2,644 groups) are excluded entirely. The prior 700 train
  and 250 validation pilot episodes are excluded. The other 5,640 eligible
  source-train groups are not selected.
- New split counts: 4,000 train / 1,000 validation / 1,000 reserved / 11,629
  excluded. Train, validation, and reserve are episode-disjoint. All selected
  IDs originate in the original source-train split; their per-episode video
  and Parquet publisher SHA-256 values are unique in the pinned inventory.
- The 1,000 new reserved-development IDs are recorded and hashed but must not
  be downloaded or loaded during this experiment.
- Exact publisher payload estimate for train+validation: **1,016,146,067
  bytes** (video 811,270,890; Parquet 204,875,177). Reserved payload is another
  206,253,249 bytes and is not included in acquisition.
- The prior task used all four windows for each selected episode. Planned
  counts are 16,000 train queries and 4,000 validation queries. Action
  normalization and the eight-prototype codebook remain train-only; no
  checkpoint selection, calibration fitting, seed search, or sealed-audit
  evaluation is planned.

The generated manifest passed its fail-closed selector validator. Its tests
cover deterministic selection, source-train-only membership, disjointness
from prior pilot IDs, reservation isolation, hash integrity, and use by the
existing hash-verifying downloader. The download smoke uses fake payloads; it
does not count as an actual source integrity check.

## Prior-manifest reconciliation and overlap gate

Two user-owned, untracked split manifests were read and left untouched:

- `manifests/pretrained_reuse/physicalai-singlearm-stack-platforms-split-20261102.json`,
  SHA-256 `69367ea49e1a261c260a80a071a181c46b523236c9fe0d10fc188b8ab429f35f`.
  Its 17,629 episode rows have the same train/validation/reserved assignments
  and world-video hashes as the tracked parent manifest, but the rows lack
  usable Parquet paths and hashes. It is not used as the source of payload
  identity.
- `manifests/pretrained_reuse/physicalai-singlearm-stack-platforms-split-v2-20261102.json`,
  SHA-256 `732758a742e46455c4bef16dccd8d9396ccede64860b2d5bdfb97447f2cba7d6`.
  Its split assignments, video/Parquet paths and hashes match the tracked
  parent manifest across all 17,629 rows; it is not an independent split.

The generated train+validation 5,000 episode selection was compared by
publisher content hashes against top-level `download-receipt.json` files in
all local `physicalai-*` artifact directories. Those receipts contained 6,374
file rows and 1,241 unique video hashes plus 1,241 unique Parquet hashes. Exact
overlap was zero for both modalities. This is an exact publisher-file check;
normalized decoded-frame overlap for the newly selected train/validation
videos remains a required pre-fit gate after acquisition.

## Local runtime and storage preflight

- C: free space: **7.37 GiB** at inspection. Acquiring the 0.947 GiB
  train+validation payload would leave about 6.42 GiB before derived features,
  above the existing 2 GiB reserve. The protected cohort is not downloaded.
- RTX 3080 Laptop GPU: 16 GiB, 0 MiB used and 0% utilization at the check.
- Python 3.11.9; PyTorch 2.6.0+cu124; CUDA 12.4; Transformers 5.6.2; PyAV
  18.1.0; PyArrow 25.0.1.
- Existing pinned DINOv2-base files are at
  `C:\CodexArtifacts\pretrained-reuse-r2\dinov2-base-runtime-smoke-01\model`.
  `model.safetensors` is 346,345,912 bytes with SHA-256
  `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`, matching
  the previous PhysicalAI feature run and the pinned revision
  `facebook/dinov2-base@f9e44c814b77203eaa57a6bdbbd535f21ede1415`. The R1
  inventory records 86,580,480 parameters and Apache-2.0 weight rights.
- The prior exact-environment feature run processed 950 episodes / 3,800
  queries in 590.96 seconds with 402,263,040 bytes peak CUDA allocation. A
  simple linear projection suggests about 52 minutes for 5,000 episodes;
  this is a planning estimate, not a runtime promise.
- Repository checks before acquisition: Ruff passed; full CPU suite **2,810
  passed, 11 skipped**; CI manifest/catalog validation commands passed.

## Acquisition attempt and recovery

The first acquisition attempt used eight workers in the intended new output
directory. It terminated with HTTP 429 for the pinned world-camera file for
episode `010750`. The process exit code was 1; no complete receipt was written.
At termination, **175 files / 17,625,486 bytes** were present (87 video and 88
Parquet files). All 175 were independently re-read and matched the new
manifest's exact byte sizes and SHA-256 values; all were selected train files.
No validation or reserved file and no `.part` file was present. The verified
files remain in the new attempt directory and are reused only after the
downloader rechecks their hashes.

Three completed-file observations were noisy (reported payload rates about
0.352, 0.068, and 0 MB/s over the observed intervals) and ended in a 429.
These were not a stable transfer-speed estimate. A cached Hugging Face token
is now read in memory and supplied only as the request Authorization header;
it is not written to receipts or logs. A CPU regression test checks this
behavior. The downloader also honors numeric/date `Retry-After` and uses
bounded exponential fallback. Focused tests pass (**6 passed**) and Ruff passes.

The two-worker resume started at 2026-11-03 01:58:40 JST in the same output
directory with `--max-retries 8`. At the 02:03:13 JST snapshot, **558 files /
56,863,546 bytes** were present and no receipt had been written. All 306 files
present at resume start were independently checked against the manifest; the
downloader verifies every newly fetched file before atomic rename. The active
request plan includes only selected train/validation files and excludes the
reserved split. The process is acquisition-only: it does not use the GPU,
extract features, fit a readout, or evaluate quality. The candidate remains
`REVIEW_NOT_ADMITTED`; the ROADMAP's unresolved scene/task gate prevents model
fitting unless independently resolved.

Three short transfer observations after resume imply roughly 0.10 MB/s and a
very rough multi-hour remaining duration at the observed point. This estimate
is noisy and not a completion commitment. No split, task, model, or reserved
partition changed; feature extraction and fitting remain unstarted.

## Execution gate and stop conditions

The intended new output directory is
`C:\CodexArtifacts\pretrained-reuse-r2\physicalai-stack-platforms-fresh-action-20261103-attempt-01`.
The existing downloader will verify every publisher size and SHA-256 against
the new manifest. After acquisition, require the complete 10,000-file
train+validation set, exact receipt/split/hash checks, zero reserved files,
and a normalized-frame cross-split overlap audit before extracting features or
fitting any readout. Use the pinned DINOv2 checkpoint and the existing frozen
feature/evaluation pipeline; do not overwrite prior outputs.

If these checks pass, the single seed-17 fit will compare the unchanged
state-only, video-only, and joint controls on the new IDs, with shuffled and
missing-input controls, per-class metrics, calibration, and episode-cluster
uncertainty. If only the joint model is better with the predeclared uncertainty
gate, report support for both-input contribution only on this same-workspace
task. Otherwise report the measured negative or uncertain result and keep the
source unadmitted. No quantization, shared-backbone work, second seed, or
product promotion is authorized by this experiment.
