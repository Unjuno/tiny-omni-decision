# MMCSG synchronized audio/video/IMU source and task screen — 2026-11-20

## Disposition

**`REVIEW_NOT_ADMITTED`; metadata only.** MMCSG is a promising source for a
speaker-attribution Decision probe because it contains aligned audio, video,
IMU and human speaker-turn annotations. It is not yet admitted for fitting:
access requires accepting a participant agreement, participant independence
across splits is not established by the published counts, and the benchmark's
native task is streaming transcription/diarization rather than a supplied-
option Decision or a proven both-input-required task. No registration,
agreement, payload, media, annotation values, or model was accessed.

## Current official metadata recorded on 2026-11-20

Checked 2026-11-20 against the publisher and challenge pages:

- Meta's [dataset page](https://ai.meta.com/datasets/mmcsg-dataset/) states
  spontaneous two-person conversations recorded with Aria glasses, audio,
  video and accelerometer/gyroscope data; human annotations include
  transcription, segmentation and labels for the glasses wearer. It describes
  the dataset as intended for research under Meta's data license.
- The linked [download and license page](https://ai.meta.com/datasets/mmcsg-downloads/)
  requires organization/affiliation details and acceptance of the MMCSG Dataset
  License Agreement before download. The agreement's grant explicitly permits
  research and commercial development/improvement of software and ML models,
  and section 2 states that resulting models may be used for academic and
  commercial purposes. It also prohibits re-identification, redistribution,
  and incorporating the dataset into another dataset or product; raw media
  must not be shipped with the model. Do not accept the agreement on another
  party's behalf. The scope of “incorporate ... into any ... product” relative
  to deployment of a derived model should be clarified with the provider before
  product-directed access.
- The official [CHiME-8 Task 3 data page](https://www.chimechallenge.org/challenges/chime8/task3/data)
  reports 172 train recordings / 8.5 h / 49 speakers, 169 development
  recordings / 8.4 h / 45 speakers, and 189 evaluation recordings / 9.4 h /
  44 speakers. Each recording has two conversation participants; audio is
  seven-channel 48 kHz, RGB is 15 fps at 720×720, and accelerometer plus
  gyroscope are sampled at 1 kHz. Human annotations provide speaker-active
  segments, transcripts, and SELF/OTHER identity. The reported speaker counts
  do not establish whether identities are disjoint between splits.
- CHiME's [challenge rules](https://www.chimechallenge.org/challenges/chime8/task3/rules)
  specify training data for system building and state that the development set
  is for evaluation, not training or automatic tuning. Respect that protocol;
  the development labels cannot become an iterative tuning set. The official
  evaluation data must remain sealed until the full candidate/protocol is
  frozen.

## Path A task fit and unresolved gates

The native labels could support a text-free binary Decision over typed
`SELF`/`OTHER` speaker options at predeclared speech intervals. This would be
a narrow speaker-attribution task, not general conversation understanding.
The official task does not itself prove that all streams are needed: the
seven-channel audio is a strong standalone source for speaker attribution, and
speaker-turn labels are not direct evidence that video or IMU improves a
decision. Any M2 claim would require matched audio-only, video-only, IMU-only,
and joint baselines plus missing, shuffled, stale, time-offset and contradictory
controls. No conclusion can be drawn from the source description alone.

Before admission or fitting:

1. Obtain written authorization to accept the agreement for the intended
   entity and written clarification that deploying a model trained on this
   dataset is compatible with the no-incorporation/product restriction.
2. Verify exact speaker/session identities and cross-split overlap from the
   official manifests without reading evaluation labels; retain recording and
   participant groups intact. The challenge's held-out evaluation remains
   sealed.
3. Define a predeclared interval-to-typed-option task, neutral candidate
   encoding, balance and deterministic sampling, and an option-/metadata-only
   prior. Determine whether any defensible task actually requires more than
   audio before a fusion fit.
4. Audit model pretraining overlap and media-level rights/consent boundaries;
   use no baseline code or weights unless independently licensed and admitted.
5. Check storage, feature-only retention and source-grouped sample size before
   any media acquisition.

**Current result:** this candidate is more directly aligned with time-aligned
audio/video/sensor Decision development than the rejected AVSD response-
generation source, but it is not presently an admitted product-quality dataset.
No model quality, multimodal benefit, or 95–98% target is claimed. Existing
artifacts and all legacy runs remain unchanged.
