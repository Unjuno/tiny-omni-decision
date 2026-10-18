# Late-object observation: fixed temporal mask-discovery pilot

The full prequantization <=200M Omni objective remains unchanged and incomplete.
Existing first-time masks improve geometric opportunity but cannot explicitly
observe later arrivals. The known train regression scene video_10903 has a hidden
item whose annotation starts at224. This scene is selected from an observed
failure, not a new blind/representative evaluation. No Decision training follows
an opportunity score automatically.

## Fixed question and input-only probe

Can the already admitted frozen SAM2.1 Tiny generate candidate masks for a later
object on the SAME32 uniform source frame IDs used by the prior feature bank?
Keep original raw native RGB, original558-frame official cut, frame IDs, checkpoint,
processor, FP32/SDPA, seed17, TF32off, deterministic algorithms and CPU4.
No target label/box/action/role chooses a frame, prompt, candidate or threshold.
All32 times and all1024 fixed32×32 cell-center positive prompts per time remain.

Keep raw binary mask proposals (logit>0), presence scores, points and boxes before
posthoc scoring. Fixed box NMS IoU0.7 retains every survivor; no TopK or confidence
selection. Empty proposals remain. No video propagation, new tracking query,
readout, adapter, encoder training, audit/dev access, download, cloud or quantization.

The installed Sam2VideoInferenceSession.reset_tracking_data API clears object
maps/prompts/outputs/history while retaining the SAME frame and vision-feature
cache. Reuse only across64-prompt chunks of one source frame, and make a new
session for each source time. No cache reuse across source frames.

Before main inference, a separate smoke process must prove:
- cached first two chunks' raw float masks/scores equal fresh-session chunks by
  shape/dtype/bytes, with independent object IDs and cleared tracking history;
- all1024 cached frame0 binary masks and scores equal the existing pinned original
  frame0 proposal archive; boxes/points/NMS indices agree;
- an encoder hook records one image encoding for16 cached chunks, versus one per
  fresh chunk; record synchronized timings, not a full-product latency claim;
- frozen state, checkpoint/processor/code hashes and CPU/CUDA RNG remain unchanged.

If parity fails, preserve the failure and do not launch the cached main path.
No tolerance relaxation or cross-frame cache assumption. Existing scripts remain
read only; new research runner/smoke/output paths are separate.

## Posthoc diagnosis, with preserved limitations

Only after all32 full proposal banks are saved, use the pinned all_train source to
score each exact grounded game-group ID and hidden-object ID. At32 source times,
report sparse-source bbox availability, interpolation distance/range, masked flags
without interpreting them as validated visible pixels, and maximum raw/NMS bbox
IoU. Interpolate bbox components only inside annotation ranges, never extrapolate;
retain missing/ambiguous/empty cases with explicit denominators.

IoU>=0.5 is a predeclared opportunity diagnostic matching the existing source
mask diagnostic, not a prediction selector, pixel-mask metric or correct-answer
count. Also record continuous maxima; do not sweep thresholds, TopK, frames or
seeds. Do not treat first-frame absence as an inference failure or count absent
source boxes as true negatives. Save per-time raw metrics and source bindings.

Even a matching mask does not select the semantic hidden item, establish its
container association, actor-relative left/right/depth, temporal identity, QA
Accuracy/NLL/Brier/ECE or genuine multimodal benefit. Poor/no opportunity remains
in the report. Keep the full four-modality95–98% target; no model promotion.

## Resource and reproducibility gates

CPU tests inspect the actual library reset/cache contract without weights/GPU.
GPU/model smoke stays outside normal CI. Local RTX3080Laptop16GB only, >=12GiB
free GPU initially,6GiB peak Torch allocation admission gate,8GiB disk reserve.
Smoke120s/main1200s admission checks between chunks/frames and at completion;
these are not external watchdogs. Save measured wall/forward/encoder counts, VRAM,
process peak, source/config/checkpoint hashes and separate new arrays. Retain every
attempt; no legacy cache/config/weight/process intervention, main merge or force push.
