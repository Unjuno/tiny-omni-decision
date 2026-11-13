# MUSIC-AVQA synchronized audiovisual source screen — 2026-10-29

## Disposition

**`REVIEW_NOT_ADMITTED`; do not acquire media, train, or use it for product
claims.** MUSIC-AVQA is structurally relevant to a bounded audio-visual
Decision probe, but its source-asset rights and split integrity do not currently
meet the project's admission gate. This is a metadata and train/validation
annotation screen only.

## Pinned observations

- Official project page: <https://gewu-lab.github.io/MUSIC-AVQA/> (observed
  2026-10-29). It says the questions cover audio-only, visual-only, and
  audio-visual scenarios, and describes temporal/spatial audio-visual
  association questions. Example descriptions include identifying the first
  sounding instrument and counting sound sources that are visible.
- Official code/annotation repository:
  <https://github.com/GeWu-Lab/MUSIC-AVQA/tree/cc8809514a2ee3d7ae81a3021c15e7c0679170d7>
  at commit `cc8809514a2ee3d7ae81a3021c15e7c0679170d7`.
- Only the pinned `json_update/avqa-train.json` and
  `json_update/avqa-val.json` contents were read, in memory. Their Git blob
  object IDs are respectively `dfb9233f598bb27ce6d2783ca377131a6ae2be2c` and
  `dcda27ddfd5de55e8e5723b5ffe5b40d35efeabc`; these are Git object IDs, not
  SHA-256 digests. The test annotation payload was not fetched or opened. GitHub
  metadata reports its size as 2,479,019 bytes and blob ID
  `84e8b4579b1bd62e5bf0b5e1e6545a3e68b39f88`.
- The train annotation has 31,927 questions over 9,277 distinct `video_id`s;
  validation has 4,568 questions over 3,815 IDs. **3,808 video IDs occur in
  both splits** (99.82% of validation IDs). This is a video-level split leak for
  any ordinary train-versus-validation evaluation. Only seven validation IDs
  are absent from train. Train/test identity overlap is **UNKNOWN** because the
  test annotations were intentionally not fetched.
- The annotation field `type` identifies modality and question category. The
  fetched train and validation files contain 17,815 and 2,549 rows marked
  `Audio-Visual`, respectively. This is a structural count, not evidence that
  the model must use both streams on each item.
- The project page reports 45,867 QA pairs, 9,288 videos, 7,423 real videos,
  and 1,867 synthetic videos. The latter two figures sum to 9,290, so the
  published corpus counts are internally inconsistent and need reconciliation.
- The page says the released dataset/benchmarks are CC BY-NC 4.0. It also says
  the performance videos were collected from YouTube and describes them as
  public there. That does not establish the rights to redistribute, download,
  process, or train on each underlying video. The code repository separately
  declares GPL-3.0; its code was not copied or executed.

## Fit to the Decision objective

The audio-visual subset is promising for a narrow, task-specific experiment:
some questions explicitly ask about correspondence between sounds and visible
instruments or require temporal association. The corpus also contains audio-
only and visual-only questions, which could provide controls. A future test
would still need to establish both-input necessity using matched audio-only,
video-only, shuffled, and cross-video controls, with video-grouped splits and
calibrated option probabilities.

The original annotation is open-ended and the project describes a vocabulary of
42 answers; it does not provide per-question candidate options. Turning it into
supplied-option Decisions would require a predeclared, type-compatible
distractor policy and validation that each constructed choice set has one
unambiguous answer. Published work also reports notable answer-distribution
bias in several question templates. No performance was evaluated here.

The official page lists 36.67 GB of real videos and 11.59 GB of synthetic
videos, plus 14.84 GB of extracted frames. The current local C: volume has only
about 0.37 GiB free. No media or extracted features were acquired. Even if
storage becomes available, the non-commercial dataset license and the
underlying YouTube rights need review against the intended product use before
any media processing.

## Gate to reconsider

Reconsider only after clip-level rights and intended-use compatibility are
resolved. If the annotation corpus is admitted for internal research, rebuild
a deterministic train/validation partition from the official train and
validation rows grouped by `video_id`; retain the untouched official test
annotations unopened until a candidate and evaluation protocol are frozen.
Then verify compatible option construction, label ambiguity, answer priors,
both-input necessity, media provenance, payload/storage requirements, and
source-group separation. The current official validation metrics must not be
used because of the measured video overlap.

Primary sources:

- [Official MUSIC-AVQA project page](https://gewu-lab.github.io/MUSIC-AVQA/)
- [Pinned official repository commit](https://github.com/GeWu-Lab/MUSIC-AVQA/tree/cc8809514a2ee3d7ae81a3021c15e7c0679170d7)
- [Pinned training annotations](https://github.com/GeWu-Lab/MUSIC-AVQA/blob/cc8809514a2ee3d7ae81a3021c15e7c0679170d7/data/json_update/avqa-train.json)
- [Pinned validation annotations](https://github.com/GeWu-Lab/MUSIC-AVQA/blob/cc8809514a2ee3d7ae81a3021c15e7c0679170d7/data/json_update/avqa-val.json)
