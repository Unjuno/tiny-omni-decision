# Fixed three-TRAIN frozen encoder probe

This is a bounded extension of the completed physical input probe, under the
user's autonomous implementation/test authorization. Preserve every old frontend,
model, feature bank, experiment and split. No training, prediction scoring,
validation/TEST/audit payload or model selection in this probe.

Input: only the three event-row0 examples in
`greatest-hits-input-physical-01/result.json`, pinned SHA
`97c91689effda1e7a080b81ddd4be7738718168493137a565f821371d493c0f6`.
Verify all18 input payload hashes before/after execution. No labels enter encoders.

- Audio: original frozen Whisper encoder8,208,384 parameters and EfficientAT
  1,961,239, same pinned revisions/checkpoint hashes as
  `native_audio_profile_v1.json`. Pass prepared8,000 samples at16kHz and16,000 at
  32kHz directly; no resampling, clipping or normalization in the encoder adapter.
  Keep the original Whisper30s processor padding and first25 output tokens.
  Require one384D event vector, frozen eval FP32 models and finite outputs.
- Video: original pinned VideoPrism114,365,184 parameters, eight existing selected
  frames. Use the historical bilinear short-side288/center-crop/float32-div255
  preprocessing, no frame repeat or full-extent policy change. Save the actual
  encoder input and2,048×768 outputs. The456×256 input becomes513×288 then crops
  left112/right113 resized pixels: only288/513 of horizontal resized extent is
  retained. This information loss remains an explicit limitation.
- Positions: save `(time relative to audio-window start)/0.5s` with each token,
  retaining video PTS for each of256 spatial tokens/frame. Use the Whisper nominal
 20ms lattice0..0.48s for25 tokens, and center0.5 for the whole-window event vector.
  Spatial coordinates use the existing16×16 grid. This documents a shared window
  clock; global Attention/receptive fields and source physical AV offset mean
  positions do not prove strict feature localization or sound/contact alignment.
- Record source commit, model/source/input hashes, original parameter counts,
  frozen state digests before/after, versions, actual forward/wall time, OS peak
  working set and audio peak allocated VRAM. No environment upgrades.
- Audio executes on idle local CUDA FP32/eager/TF32off/deterministic algorithms,
  VideoPrism in the existing isolated CPU venv. Separate processes avoid dependency
  mixing. Each phase permits900s elapsed,8GiB host peak,2GiB audio VRAM and requires
  2GiB free disk; check before load/between inputs/at end. These are cooperative
  guards, not a hard preemptive deadline for one native library call.
- Save all outputs in fresh directories. Repeat in a second process with the
  reference receipt SHA; require exact input/source pins and every saved NPY byte.
  Failure is an incomplete encoder probe, not a model quality/capacity finding.

CPU contracts first: frame-derived times instead of uniform reindexing, boundary
and malformed-input rejection, exact independent prepared-audio transport,
frozen-model refusal and expected shapes. Optional Torch tests remain optional in
normal CI. Then execute both real phases and fresh replays; preserve every failed
attempt. No gate fit is authorized by a successful forward alone. Complete source
grouping/synchronization and complementary intervention design before that fit.
