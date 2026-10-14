# Fixed native-decoder and frozen-feature container invariance probe

Purpose: test a real frozen visual frontend against a harmless container-size
change on the known training-side RoboMNIST pilot. Keep length-only classifiers
as a nuisance control; no held-out accuracy, native temporal reasoning, fusion
quality or Omni95–98% claim. No new learning or experiment selection.

## Immutable inputs and one nuisance intervention

Use `Ar1_Vhigh_Ac0_Rp01_UNC062`, the same preregistered pilot already decoded in
`robomnist-input-smoke-01`; all three native camera files. Revalidate the raw
replay02 SHA `5cd718a66f53d843ce30d9f1ba3519aa534ca977319491d90798367096576d51`
and all92 pins. The length diagnostic replay02 SHA is
`15aab2045b13c6ee64ff811d709ab77b3174658f52a0a06355354ea4e05f0182`; preserve
its84 pins and original1,475train/660validation ledger/estimators. No new source
payload or positionGT/legacy audit access. Never overwrite the original files.

Create one derivative copy per camera, preserving its complete source prefix,
then append exactly a big-endian32-bit `free` atom:8-byte header plus1,048,576
zero bytes. The [official Apple format description](https://developer.apple.com/documentation/quicktime-file-format/free_space_atom)
identifies this atom as unused container space; do not assume a decoder accepts
it without actual checks. Saved primary source text SHA
`febb055c776e072edffdfaa118c1306a3d57c4eccbd080564c92b89401ca679a`.
No reencoding, codec change, resizing, PTS/frame adjustment, alternate padding
size, camera selection, file-length tuning or retries based on predictions.

Decode all450 frames per derivative, preserving/rehashing every RGB frame and
all PTS/times/container/stream metadata. Compare against the pinned original
metadata and stored frame0/224/449 arrays. Any decoded mismatch is an incomplete
mechanics experiment, not a model-quality failure. CRC/source SHA protects
originals; derivatives have their own SHA, never the source's claimed CRC.

## Fixed real pretrained frontend and controls

Offline TinyCLIP full checkpoint:
`wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M@07a4b0bc751cb64fecd2b661c048c1dd98d69444`,
released model SHA `112bdfb07ebc8994e9afba7a2a314e278493a675033595cf48e54cf8a17ae587`.
Use existing audited model directory; all five release-file hashes must match
the R1 audit. No new model download or dependency upgrade. MIT released-weight
rights and RoboMNIST CC BY4.0 attribution remain as previously audited.
Before parsing the audit, require its historical Git LF blob SHA
`f63a3cac64d8ab84320b70368eef543742cfc54485ba656bf905f3486de984fc`
from commit `bf1530aa6f077738027fde706f1f22c320f3b1ec`. Only CRLF-to-LF
normalization is allowed for that historical check; the actual local bytes are
also pinned throughout execution. Independently require the released weight SHA
above, so an edited audit cannot authorize edited weights.

Each retained720×2560 RGB frame contains two horizontally concatenated views.
Split exactly at1280; independently letterbox each entire1280×720 lens into224²
using Pillow BICUBIC,224×126 content centered with49px top/bottom padding. Fill
with rounded255×the checkpoint image_mean. No center crop or deleted lens.
Then native processor rescale/normalize with resizing/cropping disabled and
use_fast=False. Order receiver1/2/3 ×frame0/224/449 ×left/right:18 image inputs.

CPU float32,2threads,seed17,deterministic operations,eager attention. Full
CLIPModel resident count83,101,185 includes its unused text tower; do not call it
a38M deployment. Freeze every parameter and eval mode. Encode microbatches3
with vision_model.pooler_output→visual_projection; retain unnormalized512-D
vectors as `[3,3,2,512]`, without temporal averaging. This is per-frame image
representation in temporal order, not a native learned video encoder.

Run independently through the same loaded model:
1. Original decoded sampled frames.
2. Derivative sampled frames (no copied/reused feature outputs).
3. Uniform mean-color224² tiles as a fixed evidence-destruction control.

Require original/derivative preprocessed tensors and all projected features
byte-exact; require finite features and sensitivity to the uniform control
(at least one original vector differs). Record all18 differences, not only the
largest. Fingerprint all parameters/buffers before/after, require frozen/eval/
no gradients, and CPU RNG unchanged during the forwards. CUDA must stay
uninitialized. Save/reload every tensor without lossy casts or missing keys.

Recompute camera-length/all-length posteriors from the same fixed train-only
centroids for this one pilot's original versus derivative lengths; all60 options
and stable log probabilities retained. Report both top predictions and complete
posterior change whether or not the top answer changes. Recomputed original
length-control logs must match the prior stored row within absolute1e-12, with
no relative tolerance. This is a known train
example nuisance comparison, not new Accuracy/calibration/validation or tuning.

## Resource/provenance/exit rules

New independent output root. CPU only,600s body,3GiB peak working set,128MiB
outputs and1GiB disk reserve (distinct from unchanged8GiB legacy GPU guard).
Preserve original source/model/code/protocol/library pins; pin prepared config,
rows and arrays at creation, read them back and terminal-rehash full evidence.
Resource checks happen after final readback. Exceptions save explicit incomplete
status; unavailable memory monitoring is UNKNOWN/FAIL, never fabricated success.
Commit/tests before actual execution. Save exact source commit/CLI/backend,
all resident/trainable counts and state hashes. Separate-process native/model
reload replay must verify the actual derivatives/tensors/features and controls.
No base/encoder/projector training, no Teacher artifact, paid compute, legacy
intervention, new seed, sealed evaluation, compression or main merge.
