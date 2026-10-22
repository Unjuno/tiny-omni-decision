# AMI train timestamp-to-frame join review — 2026-10-22

## Existing evidence rechecked

This is a read-only interpretation audit of the existing
`ami-hf-av-sync-check-attempt-07` artifact. It used no new media, labels,
validation/development or test content, model, or GPU.

- Dataset: `edinburghcstr/ami@46f28f2503e2ec48f8867a84eef356c70476beab`.
- Split: train only; 20 unique speech intervals, 10 speakers, 6 SA meetings.
- The sample records map speakers to camera IDs using the official pinned
  `meetings.xml`. Each interval was compared against four Closeup video streams:
  80 frame lookups total.
- Every video file was rehashed and matched the earlier pinned media preflight.
- The runner requested the midpoint timestamp of each speech interval and
  selected the first decoded video frame whose PTS was at or after that time.
  All 80 selected frames are within 0.05 seconds; recorded absolute deltas span
  0.000007 to 0.039999 seconds. This is at most one 25 fps frame period.
- Receipt SHA-256:
  `73289f21ae79ada796dbd863ddd2388b2710a72c65fe2a903796c8fd33420ab9`.
- Runner SHA-256:
  `75a29bd3df856de2879fe4acb58e42c98898bbcb77d7a570b522b93200a146c8`.
- Pinned audio row receipt SHA-256:
  `bdd5de50892445d8e3cfdc4e404a767c47a5e6a47259bee2add027317600e6b4`.
- Official meeting-to-camera mapping SHA-256:
  `8ab6cdcf03ed863e839418f31c7380392397a2eed892d73b9bc1b438c7ec520a`.
- Source video preflight SHA-256:
  `bbead20d61d57429cbf7a5358bfb5ac890e6da108a35c20e6ad8621cd67a9b4b`.

## What the result does and does not show

The `PASS_TRAIN_AUDIO_VIDEO_TIME_JOIN` status verifies that the decoder found a
frame near each requested timestamp in the selected files. It does **not**
measure an audio/video clock offset: the code chooses the first frame with PTS
greater than or equal to the requested time, so a delta below one frame period
is expected from the 25 fps frame grid. The check does not compare audio samples
against acoustic events, measure lip motion, or adjudicate who is visibly
speaking. Therefore it does not independently validate physical A/V sync or
visible-speaker truth.

This narrows the open issue: timestamp-indexed frame access is demonstrated on
20 train intervals across six meetings, but physical synchronization remains
unverified. The sample is not a frozen task set and does not establish split-wide
integrity, independent quality, or that a model benefits from both modalities.
The earlier candidate-wide visual-presence review is still only a single-reviewer
presence screen; a visible face is not proof it is the annotated speaker.

## Decision and next gate

Keep AMI at **review-only**. Do not train a quality model from this receipt. A
next train-only audit, if pursued, must independently compare the selected
speaker's headset speech timing to visible mouth activity (or another
ground-truth synchronization signal), quantify false/missing speaker visibility,
and preserve the meeting-group split. Any label adjudication needs a purpose-
specific privacy/governance review. The 111 SB windows remain exposed development
material, and no SC content may be accessed.

The source-screen branch and artifacts remain unchanged. The scope and current
status are updated in `R0_TASK_BATTERY_DRAFT_20261021.md` and `ROADMAP.md`.
