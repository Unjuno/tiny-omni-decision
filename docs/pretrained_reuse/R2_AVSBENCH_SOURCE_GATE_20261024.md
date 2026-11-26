# AVSBench source gate (2026-10-24)

## Disposition

**REVIEW / not admitted for this product's AV training or quality evaluation.**
AVSBench is a useful task lead because it defines audio-visual segmentation of
the object producing sound and includes hard cases where multiple objects are
visible but only some are sounding. It is genuinely aligned with cross-modal
evidence. However, the official dataset page labels the dataset CC BY-NC 4.0,
the processed audio/video package requires an email request, the source clips
were collected from YouTube, and its mask output is not the project's
caller-supplied option-probability contract. No data were downloaded.

## Metadata-only evidence

- Official site: <https://opennlplab.github.io/AVSBench/>. It describes the AVS
  task, the train/valid/test split, S4 and MS3 subsets, and the CC BY-NC 4.0
  dataset license. Raw-video ID CSV and masks are linked publicly, while
  processed video/audio requires an email request.
- Official code repository: <https://github.com/OpenNLPLab/AVSBench>, HEAD
  observed as `f7414aff6aedf49248887abf352471612fb1ec90`. The repository code is
  Apache-2.0; this does not replace the dataset page's separate CC BY-NC 4.0
  terms.
- The original paper reports 4,932 single-source videos (S4) and 424
  multi-source videos (MS3), with train/validation/test splits. The paper says
  videos were downloaded from YouTube with Creative Commons licenses. We have
  not audited item-level original uploader licenses, exact split IDs, duplicate
  content, or participant/source group overlap.
- AVS predicts pixel masks for sounding objects. This is a natural multimodal
  grounding task, but not a direct multiclass probability over caller-supplied
  action IDs. A valid conversion would need frozen candidate object proposals
  and a separate task contract; simple mask accuracy cannot qualify the
  requested Decision interface.
- The existing roadmap's provisional product path retains commercial-use
  compatibility as a source gate. AVSBench's NC terms do not support product
  training/deployment as currently defined. The small MS3 cohort and
  unverified group splits also make it low-power for generalization.

## Reopen conditions

Reopen only if a noncommercial research-only diagnostic is explicitly chosen,
the exact media rights and access terms are acceptable, item-level YouTube
licenses and split grouping are verified, and the segmentation-to-choice task
is frozen in advance. Such a run must remain a research diagnostic and cannot
be used to qualify a commercial product. Prefer a rights-cleared source with
native caller-option labels and enough independent video groups.

No CSV, masks, media, test labels, or model outputs were accessed. No model,
GPU, or cloud compute was used. This screen does not close R2-M-2 or qualify
audio-video product quality.

## Primary references

- [Official AVSBench project page](https://opennlplab.github.io/AVSBench/)
- [Official AVSBench repository](https://github.com/OpenNLPLab/AVSBench/tree/f7414aff6aedf49248887abf352471612fb1ec90)
- [Original ECCV 2022 paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136970378.pdf)
