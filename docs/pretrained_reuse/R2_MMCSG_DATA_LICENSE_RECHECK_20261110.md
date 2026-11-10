# MMCSG data-license and task recheck — 2026-11-10

## Disposition

`REVIEW_NOT_ADMITTED`. The public publisher materials make MMCSG a plausible
source for synchronized audio/video/IMU research, with enough published
recordings to investigate a grouped task. The official Meta download page
requires a Data License Agreement that becomes effective on first access or
acceptance. No agreement was accepted, no download/access control was used,
and no data, media, or annotations were inspected. Therefore this is a
metadata-only recheck, not dataset admission.

## Publisher evidence

- Meta describes two-party conversations with seven-channel audio, face-blurred
  video, and accelerometer/gyroscope data; annotations include transcription,
  segmentation, and labels for the glasses wearer. Meta lists ASR, activity
  detection, and speaker diarization as research tasks.
- The CHiME-8 task page reports 172 train, 169 development, and 189 evaluation
  recordings (530 total), each with audio, RGB video, and 1 kHz IMU. The
  challenge paper separately lists 49 train, 45 development, and 44 evaluation
  speakers. Thus, the 530 recording count is not 530 independent speaker
  groups; the per-split counts sum to at most 138 distinct speakers, with
  cross-split speaker overlap still unknown. Recording/session independence
  and speaker-group composition would need a permitted metadata audit.
- The CHiME-8 benchmark is streaming speaker-attributed ASR scored by
  multi-talker WER and latency. A speaker-attribution segment could be adapted
  to a supplied binary `SELF`/`OTHER` Decision, but that is a derived task, not
  the native output. Challenge rules prohibit training or automatic tuning on
  its development set. The paper says faces are blurred, notes this prevents
  many lip-motion approaches, and describes visual use as largely unexplored.
  The source therefore does not establish that both audio and video are
  necessary for the adapted Decision task. A task-specific adapter and matched
  audio-only/video-only/joint, missing, shuffled, stale, timing, and
  contradiction controls would need to be frozen before fitting.

## Data License findings and handling boundary

The [official Meta download/license page](https://ai.meta.com/datasets/mmcsg-downloads/)
states that the limited, non-exclusive, non-transferable, non-sublicensable
license permits research and commercial-purpose development/improvement of
software and machine-learning models. It also says participant-developed
models and technologies may be used for academic and commercial purposes.
However, the same agreement prohibits incorporating the dataset into another
program, dataset, or product, prohibits third-party transfer of the dataset,
and allows Meta to require deletion. The permitted model use does not clearly
settle distribution of a trained model artifact as part of this project's
product; that requires a documented interpretation before any release.

The Agreement says it is effective at the earlier of first dataset access or
acceptance. It also requires the person accepting on behalf of an employer or
entity to warrant authority to bind that entity. This agent did not accept the
Agreement or access the dataset. The archived [baseline repository](https://github.com/facebookresearch/MMCSG)
labels its **code** CC-BY-NC; that is distinct from Meta's dataset agreement,
and its implementation is not imported or reused here.

## Gate result

| Gate | Result |
|---|---|
| Public evidence for synchronized modalities | Pass at description level |
| Published recording count | 530; promising, independence unverified |
| Published speaker counts | 49 train / 45 dev / 44 eval; at most 138 distinct speakers across those counts |
| Rights for commercial-purpose model development | Expressly contemplated by Meta terms, subject to agreement |
| Agreement/access authorized for this run | **No; not accepted** |
| Product model artifact distribution rights | Unresolved |
| Native/derived Decision task | Binary speaker attribution is adaptable; it is not native option-probability output |
| Defined task requiring both audio and video | **Not established; source paper flags visual limits** |
| Official development-set tuning allowed | **No; challenge rules prohibit automatic tuning on dev** |
| Independent group split and leakage checks | Unknown; requires permitted metadata access |
| Current R2 admission | **No** |

Do not download, access the dataset page's acceptance flow, inspect challenge
payloads, or begin modeling until the user or authorized organization has
reviewed and accepted the Meta Data License and resolved model-artifact
distribution. If access is later provided, begin with a metadata-only audit of
recording IDs, participant/session groups, split membership, label availability,
and exact synchronized intervals; do not use evaluation recordings for tuning.

## Sources

- [Meta MMCSG dataset description](https://ai.meta.com/datasets/mmcsg-dataset/)
- [Meta MMCSG download and Data License Agreement](https://ai.meta.com/datasets/mmcsg-downloads/)
- [Official CHiME-8 MMCSG data description and split counts](https://www.chimechallenge.org/challenges/chime8/task3/data)
- [CHiME-8 MMCSG challenge paper](https://www.isca-archive.org/chime_2024/zmolikova24_chime.html)
- [Facebook Research baseline repository and its separate code license](https://github.com/facebookresearch/MMCSG)

