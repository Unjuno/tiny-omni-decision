# Frozen Greatest Hits audio and question features

The complete video cache continues in its original checkout. Keep its Python,
config and artifact files fixed. Implement this independent stage in the managed
`codex/r2-av-audio-cache` checkout from `2c5232d` and reuse existing local models
and environments. Never stop or restart the video process for this work.

Use the immutable 23,707 TRAIN / 8,875 development validation questions and the
same event/parent/date/asset binding. Encode each distinct event only once:
8,102 TRAIN + 3,049 validation. Compute all 24 original question/candidate pairs
without target, split, parent or option-position inputs. No new tasks, label
imputation, option reduction, resampling of examples or quality selection.

For audio, decode original denoised-media mono 96 kHz PCM with continuous PTS.
Use the existing nearest-sample half-second crop, then independent direct 96->16
and 96->32 kHz PyAV resampling. No clipping, peak normalization or extra denoising.
Reuse frozen FP32 Whisper encoder (8,208,384 parameters), original 30-second pad
and first 25 tokens; frozen FP32 EfficientAT (1,961,239) supplies one event token.
Save nominal 20 ms speech-token positions on the half-second clock and event
position 0.5. These are frontend coordinates, not proof of local token semantics
or physical AV sync. Do not create invented video timestamps to obtain them.

Reuse the pinned frozen DeBERTa pair encoder (70,830,722) and original question /
candidate string layout; save all 24 vectors and their content-hash mapping.
Model revisions, source/checkpoint/processor hashes and runtime versions are fixed
to the previous audited native encoders. All module evaluation/frozen flags and
before/after state hashes must hold. Eager attention, seed 17, FP32, TF32 off,
deterministic PyTorch, no dropout/optimizer. No environment upgrade.

First smoke the same 21 TRAIN review events: real decoded PCM must exactly equal
the saved native review PCM. Where review events coincide with the previous
three-event audio probe, compare direct resampled inputs and features exactly.
Also encode all 24 query vectors. Fresh-process replay must reproduce all outputs.
After both pass, full extraction requires the passing smoke/replay receipts and
all fixed source hashes. Record complete event/question/parent counts. The frozen
development metadata is read for binding in smoke; validation media is used only
in full extraction/replay. Official TEST, sealed audit and calibration stay closed.

Use separate fresh output directories; save per-parent audio banks plus event
mapping/times/input digests and fixed position/query banks. Reject nonfinite or
wrong-shaped features and duplicate writes. Require exact serialization and
reference equality in replay. A partial cache stays incomplete. No strict training
resume or whole-cache completion claim from partial files/status alone.

Use the idle local RTX 3080 Laptop only, while the other process uses CPU. Measure
actual wall/forward times, VRAM, host memory and any concurrent load. Fixed guards:
15 minutes for smoke, 2 hours for full/replay; 3 GiB peak GPU allocation, 6 GiB host
process working set, 16 GiB disk reserve. Check before start and between events.
No cloud cost. Stop this new process on integrity, numeric or resource failures;
do not change the running video job. Save unsuccessful attempts too.

These are deterministic feature computations, not fitted or calibrated predictions.
After complete common features, validate explicit physical event-to-feature and
candidate binding in the Decision readout. Freeze equal learning/selection rules
before Attention vs exchange/gate comparisons. The strict complementary-task and
four-modality 95–98% product requirements remain unchanged and unmet.
