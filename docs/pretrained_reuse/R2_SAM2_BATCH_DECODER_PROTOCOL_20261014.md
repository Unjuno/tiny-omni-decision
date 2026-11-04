# R2 SAM2 cached decoder batch probe — fixed 2026-10-14

## Purpose and scope

Wider input-only cue/cover training requires more than the known one-scene
diagnostic. Native per-time proposal discovery currently takes about9min for32
times. Before expanding that cost, measure whether its independent initial-frame
mask decoder can process64 prompts together with the same frozen native weights.
This engineering probe is not a new training/QA/semantic relation experiment.

Use known **train** video10903, frame0 and first64 points of the existing32×32
native RGB cell-center grid. This prefix is selected for its existing saved raw
reference, not annotation quality. Original raw-prefix SHA-256:
`73ebefe25631de792d0bc4f10e6492801b88383b12d49354c6bbbbbe045a901c`.
Original cache-smoke result:
`bc05e706e702013b36a123166077e092442966f2e29ab01d785792bc55c83684`.
No annotations, dev/sealed data, new weights, download, learning or environment
upgrade. All old model/config/data/proposals/checkpoints remain read-only.

## Bounded decoder change

One unchanged `Sam2VideoModel`,38,962,498 registered parameters, strict native
load, FP32/sdpa/TF32off/deterministic seed17/CPU4. Use the existing native frame
preprocessing, exact session-scaled prompts and cached native image features
with the unchanged initial-frame no-memory embedding. No inter-frame cache reuse.
The new helper calls native prompt encoder/mask decoder as one image with64
independent single positive points, preserves best-IoU multimask selection and
native no-object score<=0 → mask=-1024 semantics. Keep all objects. Skip only
unused1024-resolution interpolation, object pointers and temporal memory outputs.
Do not batch through the high-resolution wrapper's incompatible shape assumptions.

CPU tests use hand-checked gather/presence/tie/permutation cases and the actual
installed tiny randomly initialized native heads against serial native low-res
outputs. The tiny test forces positive presence so an all-absent mask comparison
cannot pass vacuously. No pretrained checkpoint or GPU is used in CPU tests;
Torch/Transformers tests skip when optional packages are absent in normal CI.

## Fixed actual measurement and admission

Native first call must match archived64 raw FP32 masks and presence scores
byte-for-byte. Warm native and batch paths, then alternating two matched64-object
forward pairs: native→batch, batch→native. Both reuse the same frame embeddings;
no image encoder call in measured windows. Setup/preprocessing and output D2H
conversion are outside these forward windows; record full body separately.
This is a cached proposal-forward comparison, not end-to-end product latency.

Record byte equality independently from these predeclared numerical guards:
raw-mask max absolute difference≤1e-3, score difference≤1e-5, relative tolerance0.
These FP32 batch guards do not permit changed dataset outputs: admission ALSO
requires exact all64 threshold0 binary masks, normalized boxes, own-foreground
pixel seeds/nonempty flags and prefix NMS identities under existing IoU0.7.
The64-point NMS is not full1024-grid NMS. No tolerance, point, seed, policy or
threshold change after observing outputs. Preserve differences and classify
completed-but-not-drop-in separately from failed/incomplete execution.

Candidate warmup/two measured calls must repeat exactly; both measured native
calls must match the archived prefix. Check frozen weights/None gradients/model
hash, unchanged CPU/CUDA RNG, embeddings/prompts/preprocessed pixels and all
source hashes. Save all8 raw/score arrays; fresh process with identical contract
must reproduce shape/dtype/bytes. Record call order/windows, encoder counts,
global peak allocation/reserve, GPU/library versions, body time and hashes.

Fresh output directories, GPU free≥12GiB, disk reserve≥8GiB+128MiB. Gate body≤180s
and peak allocation≤8GiB after calls (not watchdogs). OOM/numerical/source faults
preserve incomplete attempt; no retries with silently changed settings. No live
process intervention, cloud spending, compression or main merge.

## Next decision

Passing64-prefix checks permits full-grid/multiple-fixed-time admission only;
it is not permission to replace the frozen scientific corpus or claim wider
quality/latency. If any discrete output differs, reject drop-in substitution and
retain the measured speed/differences; any separate derived-corpus proposal needs
its own fixed input/quality protocol. Existing nearest-distance relation policy
stays closed. Semantically supervised cue-to-cover/identity memory and useful
native multimodal Decision quality remain open.
