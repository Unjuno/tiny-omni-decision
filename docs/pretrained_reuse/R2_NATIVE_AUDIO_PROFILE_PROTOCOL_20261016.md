# Common native audio profile: bounded two-source smoke

The full language cache stopped before model loading because remaining C: space
could not hold974,550,528tensor bytes plus the fixed2GiB reserve. Preserve
`native-language-cache-01` as INCOMPLETE. No resource threshold was relaxed.

Continue the independent audio input unit without allocating a full corpus bank.
Reuse the actual pinned speech/event encoder setup from the native integration.
No optimizer, scoring, checkpoint selection, validation or audit access.

- Original Whisper encoder8,208,384 parameters; EfficientAT1,961,239; mel0.
- FP32,CUDA,eager Whisper,batch1,TF32off,eval/frozen,seed17; same installed packages.
- Fixed existing TRAIN Lombard `s11_l_bbab2n`: mono16k,35,200samples,110speech tokens.
  Require exact speech AND event feature parity with saved native integration.
- Fixed first TRAIN STARSS window `fold3_room12_mix001:t0000-0005`: original
  five-second24k FOA PCM channel0, as in the existing event feature profile.
  Retain source hash/window identity. No averaging/inference of channels.
- Speech uses original16k waveform or CPU sinc24k->16k, official30s processor
  padding, and only ceil(physical16k samples/320) tokens. Event features resample
  original16k/24k directly to32k on the model device; no24->16->32 two-hop change.
- No amplitude normalization, transcript, labels, cropping or frame changes.
  STARSS event encoding now uses the native CUDA profile; do not assert bit parity
  with its historical CPU bank. Its new speech path has no old feature reference.
- Reject unsupported rate/channel/duration, nonfinite values, unfrozen models or
  mismatched output shapes. Save all six PCM/feature arrays and read them back.
- New directories `native-audio-profile-smoke-01` then `native-audio-profile-replay-01`.
  Fresh replay recomputes both inputs and compares all six arrays exactly.
- Require clean committed source, original source revisions/input hashes before
  and after, and frozen model-state equality. Keep input/config/source/environment
  and output hashes. Resource acceptance:300s,2GiB peak CUDA allocation,4GiB OS
  peak working set,2GiB disk reserve checked before load and at completion.

This smoke implements the common preprocessing primitive. Full train/validation
audio extraction and the shared decision fit remain separate unfinished work.
Capacity must be available before complete persistent caches are allocated.
Following source changes, any language retry needs a new same-source smoke and
new output directory; it must not reuse the failed partial cache as complete.
