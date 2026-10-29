# RoboHEARD 2027 synchronized AV source screen — 2026-10-29

## Disposition

**Review-only lead; not admitted for acquisition, fitting, or product claims.**
This is a promising task-aligned source for multi-speaker audio-visual speech
recognition (AVSR), but it is not yet a frozen-feature Decision task and its
data terms are non-commercial. Do not register, accept an agreement, or fetch
training/development/test payloads as part of this screen.

## Pinned observation

- Official page: <https://roboheard.github.io/>.
- Observed 2026-10-29 (Asia/Tokyo). The page is a live GitHub Pages site with no
  source revision exposed in the inspected page; therefore the upstream source
  revision is **UNKNOWN**. Preserve the captured-page SHA-256 below as the
  observation identity, not as an upstream commit.
- Three in-memory GETs of the 40,771-byte official HTML page returned the same
  SHA-256 `ac739eed17c91ab621a95257426f964f7ddf35317658733d88905d0bfd685326`.
  Elapsed times were 0.475 s, 0.024 s, and 0.023 s (arithmetic mean 0.174 s).
  These requests measure only retrieval of a small metadata page; they do not
  estimate dataset throughput or a media-download wait time. No dataset media,
  annotations, or model weights were fetched.

## What the official page establishes

The challenge describes over 100 conversations, collected with QUANTA X2 (or a
newer robot) in home-companionship and navigation/shopping scenarios. Sessions
are approximately 5–8 minutes with 3–6 participants. It reports about 60 hours
of audio split into 30 h train, 10 h development, and 20 h test, with speaker-
and environment-disjoint partitions. Inputs are multi-channel audio (2- or
6-microphone configurations) and multi-view cameras. The stated task is to
identify who spoke, when, and what they said; the offline and online tracks are
scored by time-constrained minimum-permutation character error rate (tcpCER).

The page states CC BY-NC-SA 4.0 for the data. It also says that participants
must register and declare datasets/checkpoints to organizers by 2026-11-30;
the training-data links were announced for 2026-10-23 and the development data
and baseline for 2026-10-30. No registration was submitted and no access terms
were accepted. The page reports the split policy but does not publish exact
recording, speaker, or environment IDs in the inspected page.

Primary source: [RoboHEARD 2027 official challenge page](https://roboheard.github.io/)
(live page observed 2026-10-29; captured HTML SHA-256 above).

## Relevance and unresolved gates

This is closer to the intended audio/video/sensor decision setting than an
ordinary visual-only dataset: it has real robot-mounted sensors, moving people,
overlapping speech, and a task whose stated output associates speech with
speakers and time. It could support a task-specific structured Decision output
after the exact label format is inspected. It does **not** by itself establish
that each example requires both audio and video, nor that the task is equivalent
to calibrated probabilities over supplied options. Audio-only, video-only,
cross-modal-mismatch, and source-grouped controls would be needed on the
official development partition before any quality claim.

The CC BY-NC-SA 4.0 restriction creates a product-use uncertainty for a model
intended to remain a general product candidate. The exact payload size,
participant consent/provenance details, annotation schema, group IDs, and
precise access conditions are also unknown. The official page's statement of
speaker/environment-disjoint splits is promising but is not yet an independently
checked split manifest. The local C: drive had about 0.372 GiB free and the RTX
3080 Laptop GPU was occupied during this screen, so no media acquisition or GPU
work was appropriate.

## Gate to reconsider

Reconsider only after the official development release is available and its
access terms can be reviewed without submitting a registration on the user's
behalf. Before fitting, inspect a pinned manifest and sample schema; verify
recording/speaker/environment disjointness; establish whether labels support a
bounded supplied-option decision and whether both modalities are necessary;
confirm derivative-model/product-use compatibility; estimate actual payload
size and storage needs; then preregister train/development controls and metrics.
Until those gates pass, retain `REVIEW_NOT_ADMITTED` and continue with CPU-only
work that does not expose any official test data.
