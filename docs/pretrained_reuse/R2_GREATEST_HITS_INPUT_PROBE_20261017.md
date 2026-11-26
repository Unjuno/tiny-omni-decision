# Fixed native input probe after the development-media audit

Existing `native_audio.speech_waveform` admits16/24kHz only. The physically inspected
Greatest Hits source is mono96kHz. Preserve the old frontend and its hashes. For
the new source, prepare independent96→16kHz speech and96→32kHz event inputs; never
route the event branch through16kHz, which would remove information above8kHz.
Use the already-installed PyAV/FFmpeg mono float resampler, including final flush;
no peak normalization or clipping. Preserve the original96kHz samples and hashes.

This bounded CPU input probe loads no encoder or optimizer and evaluates no model.
It follows the full409-parent integrity audit; do not execute before that audit
passes. Use the first3 TRAIN parents sorted by SHA256(`17:`+ID), **event row0** in
each regardless of missing labels. These are fixed IDs:

-2015-03-30-00-26-27
-2015-03-12-20-15-29
-2015-03-31-02-08-23

For each event, center a48,000-sample window on the nearest96kHz sample to the
released onset (Python round, ties-to-even), recording the<=half-sample timing
error. Require the full0.5s interval within source PCM; do not pad/truncate.
Audio samples must have finite float32 values, one channel,96kHz and continuous
PTS. Decode the original media without retiming.

Use the actual video PTS inside the same sampled audio interval, selecting8 distinct
frames at evenly spaced indices across that interval. Fail if fewer than8 frames;
record their actual timestamps and offsets. Keep raw RGB uint8 frames and a contact
preview; do not choose spatial crop/encoder normalization in this input-only stage.
The source window can contain motion/sounds beyond the labeled impact; the task
uses supplied event localization and does not claim learned localization.

Tests first: exact audio crop/timing error, boundary refusal, eight distinct PTS
frames, unknown/malformed input refusal. In a separate CPU process verify native
PyAV conversion length, finite outputs and exact repeatability with a fixed0.5s
4kHz+12kHz synthetic signal: the16kHz branch rejects the12kHz component while the
32kHz branch retains it. This is a resampler contract, not model quality.

After completed media audit, run the fixed three-parent physical input probe and
save original/cropped/resampled/frame hashes, source revision, all timings, decoder
versions and exact save/reload checks in a fresh output. Do not alter the existing
common corpus, checkpoints or validation. Complete grouping/physical synchronization/
semantic intervention gaps before any conditional-gate quality claim or training.
