# CaptainCook4D media and group recheck — 2026-10-21

## Decision

Keep CaptainCook4D as a **provisional text+video task candidate**, but do not
start a quality fit from the currently available material. A single previously
acquired training recording supports a low-cost frame-decoding smoke. The
current official links failed for five sampled training GoPro streams and two
sampled synchronized HoloLens PV streams. In addition, the official training
split contains only six participant groups, heavily imbalanced by participant;
the official validation split has only two participant groups. This is too weak
to support a high-confidence generalization or 95% product-quality claim.

The candidate task remains: given a step description and its observed video
segment, return probabilities for `consistent` versus `inconsistent` using the
published `has_errors` label. This is a plausible text+video Decision task, not
evidence that audio is required. No model was loaded, trained, or evaluated.

## Pinned source and read-only checks

- Annotation source: `CaptainCook4D/annotations@a8a920a3293c4db27099a20ddbe3a3a9be1283e3`.
- Downloader metadata: `CaptainCook4D/downloader@cf248fe2139fc6e37f9737141ec719948fbd1747`.
- Official project metadata reports Apache-2.0 data terms; the task must still
  honor the source's consent and release conditions.
- Official train has 204 recordings. Removing `15_33`, the sole recording
  sharing a participant with official validation, leaves 203 recordings and
  2,956 labeled step rows (1,045 positive error labels) across six participants:

| Participant | Recordings | Labeled steps | Error labels |
|---:|---:|---:|---:|
| 1 | 3 | 50 | 37 |
| 2 | 1 | 12 | 9 |
| 3 | 55 | 801 | 282 |
| 4 | 55 | 810 | 270 |
| 5 | 43 | 600 | 230 |
| 6 | 46 | 683 | 217 |

The previously recorded official validation cohort has 84 recordings from two
participants after resolving the single train/validation person overlap. The
official split therefore cannot by itself support a precise participant-level
quality estimate. A participant-held-out cross-validation entirely within the
six-person train set could only be exploratory and must report six independent
groups, not thousands of steps as independent examples.

Read-only `HEAD` checks against the pinned downloader URLs returned HTTP 404 for
the `gopro_360p` streams of training recordings `12_10`, `10_46`, `1_10`, and
`13_45`. Earlier same-day checks recorded 404 for `10_16` and validation `10_31`.
Follow-up `HEAD` checks also returned 404 for `hololens_sync_pv_video` on `10_16`
and `10_31`. No bytes were downloaded during these checks.

One old, already-present training file remains at
`C:\CodexArtifacts\pretrained-reuse-r2\captaincook-media-probe-01\10_16.mp4`.
Its SHA-256 is
`c243764b1937aa5cdca29c758dde0cb8f4c7bf24bce84994d0229ef97657ce80` and its
size is 104,842,046 bytes. The saved October 13 response records HTTP 200 from
the official `gopro_360p` URL at that time. On this recheck, PyAV 18.1.0 decoded
RGB frames at 0%, 50%, and 90% of its 973.639-second video stream (640×360); the
CPU operation took 0.311 seconds. This is a train-input decoding smoke only.
The file was not modified or copied.

## Split-access caveat

The ad hoc group-count script loaded the combined split JSON into process memory
and selected only its `train` key; that JSON also contains official test
recording IDs. No test annotations, test media, test metrics, or model outputs
were loaded or evaluated, and no test IDs were emitted or retained. Because
the membership metadata was technically parsed, this recheck does not claim
that official test IDs were never read. No sealed-audit data was accessed.

## Next gate

Before any candidate training, require working publisher-authorized media for
enough participant groups and freeze a new participant-held-out development
split using training recordings only. If adequate group coverage cannot be
obtained, leave CaptainCook4D as a data-plumbing smoke and continue the R2 source
search. Do not use the official test split to compensate for the small train or
validation group counts.

No live process was touched; no GPU, cloud compute, optimizer update, checkpoint
or dataset split artifact was changed.
