# R2 HoloAssist temporal readout results — 2026-10-19

## Scope and result

This is a development-only test of a small causal decision head for the fixed
HoloAssist task: given the task instruction and the synchronized recent video
and audio, classify whether each annotated next action is correct. It is not a
general Omni benchmark, a product model, or evidence for the 95–98% target.
Text is present in every arm as the task-question embedding; `video_only` and
`audio_only` mask the other sensor, while `joint` uses both sensors and text.
All three encoders are frozen. Only the 379,652-parameter gated LSTM readout is
trained. The validation set is session/media-disjoint and the test split was
not loaded.

The complete fixed budget ran to 2,048 updates for each arm. Best checkpoint
selection used validation NLL, with no early stopping. All best-checkpoint
reload checks passed exactly (maximum probability difference 0). The best
checkpoints occurred early, followed by large NLL and calibration degradation.
At selected checkpoints, audio-only was strongest; joint audiovisual input did
not beat it. This run therefore does not establish a useful fusion gain, and
the full 2,048-step endpoint is markedly overfit.

| Arm (task text always present) | Best step | Best Accuracy | Best NLL | Best Brier | Best ECE | Final Accuracy | Final NLL | Final ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Video only | 192 | 0.7131 | 0.5844 | 0.3951 | 0.0170 | 0.6656 | 2.5743 | 0.2985 |
| Audio only | 320 | **0.7379** | **0.5295** | **0.3516** | 0.0346 | 0.7073 | 2.0757 | 0.2588 |
| Joint audio/video | 128 | 0.7305 | 0.5474 | 0.3650 | **0.0301** | 0.7173 | 2.4093 | 0.2607 |

The validation set contains 1,896 labelled windows from 207 distinct sessions
and media assets, plus 81 unlabelled context windows. Joint accuracy is 0.74
percentage points below audio-only at each arm's independently selected
checkpoint; the NLL gap is 0.0179. The arms select different checkpoints, so
this is a practical validation comparison, not a paired same-step modality
causal estimate. Per-task joint metrics are in the machine-readable result.

## Training curve and overfit evidence

Each arm used seed 17, batch size 8 sessions, AdamW at 0.001, and the frozen
2,048-step config. The 160 training sessions contain 1,550 windows, including
unlabelled causal context. Each arm made 16,384 session presentations, covering
all 160 unique sessions and all 1,550 unique windows, but revisiting windows
157,140 times. Rolling train metrics in the raw history are sampled during
training and are not treated as a fixed-set generalization gap.

For example, video-only validation NLL improved from 0.6453 at step 0 to its
0.5844 minimum at step 192, then rose to 0.7123 at 512, 1.5670 at 1,024, and
2.5743 at 2,048. Its ECE rose from 0.0170 at the selected step to 0.2985 at the
endpoint while mean confidence reached 0.9641. Audio-only NLL reached 0.5295
at step 320 and ended at 2.0757; joint NLL reached 0.5474 at step 128 and ended
at 2.4093. Their final mean confidences were 0.9661 and 0.9776. This is clear
validation overfit under the fixed schedule, not a reason to hide the endpoint
or extend the run.

The joint best-checkpoint accuracy/NLL by task type (counts differ across
tasks) was:

| Task type | n | Accuracy | NLL |
|---|---:|---:|---:|
| Assemble computer | 77 | 0.8442 | 0.4548 |
| Assemble laser scanner | 43 | 0.8837 | 0.3650 |
| Assemble nightstand | 230 | 0.7696 | 0.4799 |
| Assemble stool | 107 | 0.7944 | 0.4856 |
| Assemble tray table | 106 | 0.7170 | 0.5641 |
| Assemble utility cart | 236 | 0.6949 | 0.5934 |
| Change belt | 4 | 1.0000 | 0.1330 |
| Change circuit breaker | 18 | 0.6667 | 0.7649 |
| Disassemble nightstand | 148 | 0.7568 | 0.5559 |
| Disassemble stool | 79 | 0.7722 | 0.5296 |
| Disassemble tray table | 64 | 0.6250 | 0.5950 |
| Disassemble utility cart | 138 | 0.8333 | 0.4548 |
| Fix motorcycle | 25 | 0.6000 | 0.6628 |
| Make coffee with espresso machine | 31 | 0.7742 | 0.4530 |
| Make coffee with Nespresso machine | 57 | 0.7895 | 0.5327 |
| Setup big printer | 57 | 0.4912 | 0.8200 |
| Setup camera | 82 | 0.5854 | 0.6477 |
| Setup GoPro | 160 | 0.5875 | 0.6790 |
| Setup small printer | 87 | 0.7126 | 0.5745 |
| Setup switch | 147 | 0.8163 | 0.4505 |

The four-example “change belt” score is not meaningful evidence by itself.
Task heterogeneity is substantial; e.g. setup big printer and setup GoPro remain
weak despite stronger assembly categories.

## Data, feature banks, and execution record

- Frozen windows: 1,550 train windows / 160 sessions; 1,977 validation windows
  / 207 sessions. Train/validation session and media overlap: zero. Their
  window receipt hashes are `52e028767aa00c5d2effc2735a1a404d1a18a64a7ee1a39d947e1c59f7041b81`
  (train) and `b58c198d611ceab8d2d435004203ef8f5079a389e3560a41937118480c64c671`
  (validation).
- Text: 3,527 x 384, MiniLM frozen bank, SHA-256
  `53bbce5ef42345dbe7431fd2369766716b16e51105f8f4adf5208b523492f9a3`.
- Audio: 3,527 x 384, Whisper Tiny frozen bank, SHA-256
  `0eafa0a03898d52647f8b8bf7895564e488c41a826cdd01167b16dcf4e4a13c4`.
- Video: 3,527 x 768, VideoPrism frozen bank, SHA-256
  `a67abc38d733d54bc875ac2cc5aead43c62ec112ceafa3c815cd5799cba48102`.
- The video extractor processed eight causal frames per window and completed
  on CPU/JAX in 10,153.17 seconds (about 2 h 49 min); peak RSS was
  4,168,560,640 bytes. Text extraction took 2.55 seconds on CPU; audio
  extraction took 444.63 seconds on the local RTX 3080 through HotUGPU.
- The first cache-pack attempt caught a real implementation mismatch: the
  extractor correctly stores one ordered train+validation bank, but the packer
  passed that full ID list to a split-level equality check. Fixed by selecting
  each split in frozen window order after validating the whole bank against the
  concatenated windows. Four focused tests and the full CPU suite passed after
  the fix.
- Readout training ran on the local RTX 3080 Laptop GPU, Python 3.11.9,
  PyTorch 2.6.0+cu124. Total arm time was 157.55 seconds. Per-arm CUDA peak
  allocated/reserved memory was 87,205,376 / 102,760,448 bytes. HotUGPU
  reserved 3,200 MiB; Orbitune briefly resumed concurrently during the run,
  with system GPU memory peaking near 16.2/16.4 GiB. No OOM occurred and no
  Orbitune process or model was modified. Cloud compute/cost: none.

Run source commit: `327d63e61b279e293520e7afaef3f22294ff214b`. Config SHA-256:
`f4de89195cbe5928a65a7b0ca634ee86ea3e25548856db07fbeca020d307ac32`.
Packed train/validation feature hashes are
`1239b5c77c3c44f8d62e26aa0231e70778f41fd17d6bbdfee5465d26c2845999` and
`6a80d8aad25faada65039cf2586e28f2dceeed97ba5aee6bea81d5e16c15cae8`; cache
receipt SHA-256 is
`050420469aad527fb4a61e7cdb46ae044cc0e694cae729b4877faf252f95d2c5`.
Validation sample-ID hash:
`17d62b298bc1b5f2eceff362e7c66da0f75f4150ae87aef2ecd85ef79eca9dd2`.
Effective config, selected-step learning curves, per-task metrics, prediction
and checkpoint hashes are preserved in
[`R2_HOLOASSIST_TEMPORAL_READOUT_RESULTS_20261019.json`](R2_HOLOASSIST_TEMPORAL_READOUT_RESULTS_20261019.json).
Full predictions, histories, and checkpoints remain outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\holoassist-video-pilot-160-207-seed17-pitchshifted-attempt-01\readout-attempt-01`;
the packed cache is in sibling `packed-cache-v1`. Existing Teacher/v1/v2
artifacts were not changed. No test data or sealed audit was opened.

## Conclusion

**Run execution: PASS.** All three arms completed 2,048 updates, the cache and
split checks passed, and selected-checkpoint reload matched predictions
exactly. **Fusion benefit: not supported by this pilot.** The joint arm did
not outperform audio-only at validation-selected checkpoints, and all arms
showed severe late overconfidence and validation-NLL regression. The useful
next action is to treat this as a narrow task/data/optimization failure signal:
inspect label quality and within-session temporal target alignment on the
training/validation materials, then design a prospectively distinct task test
before another fit. Do not continue tuning these same validation IDs, and do
not promote or use this result to claim broad four-modality quality.
