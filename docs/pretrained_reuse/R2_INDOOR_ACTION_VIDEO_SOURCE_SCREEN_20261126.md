# Indoor Action video source screen — 2026-11-26

## Disposition

**`REVIEW_NOT_ADMITTED`; metadata-only.** The task shape is relevant to a
video-only action Decision, but the small participant pool, person overlap
across splits, a publisher count inconsistency, and unresolved consent scope
make it unsuitable for a general video-quality claim. No archive, labels,
media, model, or sealed audit were accessed. No fitting or evaluation was run.

## Pinned source evidence

- University of Granada record: DOI `10.5281/zenodo.14551902`, version 1,
  open access; the record metadata declares `CC-BY-4.0`.
- Official download script repository: `DaniDeniz/IndoorActionDataset`,
  inspected commit `89b896bba5d6c9b199d927129af0e4205605572d`.
- The repository itself displays BSD-3-Clause. The dataset record's explicit
  CC BY 4.0 declaration is the evidence used for dataset reuse; the code
  license is not substituted for the data license.
- Official record file inventory reports
  `IndoorActionDataset-video.zip`: 977,124,409 bytes, MD5
  `5f5d438335692b32127b0f780d98236f`. An HTTP HEAD against the publisher's
  linked download endpoint returned the same byte length and `application/zip`;
  no response body was requested.

## Task and split limits

The publisher describes 1,228 labeled clips across ten action/no-action classes
from five people, with each clip traceable to a longer source video. It says all
clips from an original video are assigned to the same train, validation, or test
split, which is a useful recording-level grouping rule for a video-only
classification Decision.

The source's own tables do not reconcile fully: class totals sum to 1,228 and
split totals of 673 train / 187 validation / 368 test, while the subject table
has a six-clip discrepancy for S2 and sums to 362 test clips. The publisher
reports subjects in multiple splits, so these partitions do not test
person-held-out generalization. Five participants are too few to support the
project's broad video-quality target.

## Rights and privacy limits

The Zenodo record declares CC BY 4.0 for the dataset. The source describes
identifiable people performing activities in bedrooms, kitchens, living rooms,
and other indoor settings. The official record, repository README, and linked
paper were checked for a consent / participant-release statement; none was
found in those reviewed materials. This does not establish that consent was
absent, but downstream model-training and privacy scope remain unverified.

The source's paper reports conventional action-recognition experiments; those
are not evidence for this project's calibrated supplied-option Decision
interface or for multimodal fusion. If the rights and privacy scope are later
clarified, a valid probe would still need exact source-video groups, class
support by split, a frozen train/validation model-selection protocol, and one
test evaluation after configuration freeze. It should be described only as
within-participant, source-video-held-out action classification.

## Decision

Do not download or fit this source under the current evidence. The potential
video-only task does not resolve the open joint-modality gate and has only five
participants. Keep the candidate at `REVIEW_NOT_ADMITTED` unless the data
provider clarifies participant rights and a narrowly scoped research use is
approved under the project's data policy.

## Primary sources

- [University of Granada dataset record (DOI 10.5281/zenodo.14551901)](https://produccioncientifica.ugr.es/documentos/67a9c7ca19544708f8c72ad5?lang=en)
- [Zenodo version 1 record (DOI 10.5281/zenodo.14551902)](https://zenodo.org/records/14551902)
- [Dataset repository](https://github.com/DaniDeniz/IndoorActionDataset/tree/89b896bba5d6c9b199d927129af0e4205605572d)
- [Pinned download script](https://github.com/DaniDeniz/IndoorActionDataset/blob/89b896bba5d6c9b199d927129af0e4205605572d/setup_video_data.sh)
- [Associated paper](https://doi.org/10.3390/electronics13234786)

## Machine-readable record

See [`indoor-action-video-source-screen-20261126.json`](../../manifests/pretrained_reuse/indoor-action-video-source-screen-20261126.json).
