# XRFv2 Plus source screen — 2026-10-29

## Disposition

`REVIEW_NOT_ADMITTED_FOR_PRODUCT_WORK`. XRFv2 Plus is structurally relevant
to sensor/video decision tasks, but the official Kaggle API reports the
dataset license as `Unknown`; Zenodo's CC BY 4.0 metadata applies to a record
whose only attached file is a 1.15 MB preprint. The author's MIT-licensed
GitHub repository licenses its code, not demonstrably the Kaggle data. Rights
for the dataset payload therefore remain unclear. Do not download or use the
payload until the dataset license is explicitly established. No files or
annotations were downloaded/opened and no access request was submitted.

## Pinned source evidence

- Zenodo record [10.5281/zenodo.20564312](https://zenodo.org/records/20564312)
  describes 853 synchronized sequences, 16 volunteers, 3 scenes, 34 action
  classes, 8,157 segments, and 108,929 QA pairs. It advertises CC BY 4.0 in
  metadata, but its only attached file is `main.pdf` (1,152,780 bytes).
- The official project repository is
  [airslab2020/XRFV2](https://github.com/airslab2020/XRFV2). Its README links
  the Kaggle release, says the release is version 4, and describes Wi-Fi CSI,
  five-position IMU, AirPods IMU, depth/IR video, video embeddings, pose/mesh,
  and action annotations. It documents an MIT license for the repository;
  that statement does not establish rights for the separately hosted dataset.
- The official Kaggle dataset API endpoint is
  `https://www.kaggle.com/api/v1/datasets/view/airslab2020/xrfv2-multimodal-tal-caption-qa-no-rgb`.
  It reports `licenseName=Unknown`, public status, version 4, and
  `totalBytes=210794980577` (about 210.8 GB decimal / 196.3 GiB). The release
  description says raw Kinect and cropped RGB videos are excluded for privacy.
- Kaggle's API description lists 853 samples and multiple sensor/video
  representations plus English action QA. It does not establish that a
  specific decision target requires both modalities, nor does the metadata
  provide an independently grouped train/validation/audit split.

## R2 relevance and gates

- **Candidate task:** action/temporal decision from synchronized sensor and
  video streams; task fit is plausible, not validated.
- **Both-input necessity:** `UNKNOWN_NOT_TESTED`.
- **Independent groups and splits:** volunteer IDs and scenes are described,
  but a held-out volunteer/scene split manifest was not established from the
  metadata. Must be verified before any evaluation.
- **Rights:** `UNRESOLVED`; Kaggle says `Unknown`. Zenodo CC BY and GitHub MIT
  do not by themselves clear the separately hosted data.
- **Storage:** roughly 196.3 GiB, incompatible with current local disk reserve.
- **Use performed:** fetched public record/API metadata only. No dataset
  payload, annotations, or media; no fitting, evaluation, or sealed audit.

## Next action

Seek an explicit dataset license from the provider and verify subject-grouped
splits before reconsidering. Do not use the model-code MIT license as a proxy
for data rights. Continue R2-M source discovery without downloading this
payload.
