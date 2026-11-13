# CREMA-D synchronized AV source screen (2026-10-11)

## Decision

Do not add CREMA-D to the approved Path A corpus or fetch its media in this
workstream. Keep it as a **review-only, task-mismatched candidate**. No dataset
files, access form, or media were downloaded or submitted.

The official repository describes 7,442 clips from 91 actors and requires
users to fill an access form. Its labels are acted emotional expressions and
crowd judgments of emotion/intensity for voice-only, face-only, and
audio-visual presentations. That is a narrow affect-recognition task; it does
not measure general audio-video event understanding or the current CLEVRER
reasoning gap. The source includes actor demographics, so this work does not
recast those labels as reliable observations of actors' actual internal
states.

The repository states ODbL 1.0 for the database and DbCL 1.0 for individual
contents. This is useful license evidence, but it does not resolve every
right or approval relevant to identifiable video and voice. The project also
has no explicit CREMA-D license allowlist entry or approved corpus manifest.
Therefore neither license text alone nor the public repository is treated as
project approval. Whole-repository Git LFS clone is listed as approximately
7.55 GB; no assets were fetched.

## If reconsidered

First decide whether a research-only acted-affect task belongs in this
product's evaluation scope and obtain/record the required dataset access and
rights review. Then pin the source revision, archive hash and exact license
files; inspect the data collection consent and any use restrictions; build a
media manifest; and split by actor identity before model selection. Keep all
questions and media from one actor in one split. Report audio-only,
video-only, and joint predictions on identical actor-held-out examples, with
modality shuffles and group-level uncertainty. Do not use CREMA-D as a proxy
for general AV quality or sensor fusion.

## Evidence checked

- Official repository at checked HEAD
  `1658cd342dff90010aa843eaeebd53610a08b1dc`:
  [README](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/README.md)
  (dataset size, acted task, access form, storage, licenses, labels and actor
  metadata).
- Original dataset paper:
  [CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/)
  (recording and rating protocol; the PMC page was CAPTCHA-gated during this
  check, so the repository's cited summary is the directly reviewed evidence).
- License texts:
  [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) and
  [DbCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/).

No model was loaded, no GPU work ran, and no change was made to existing
datasets, manifests, checkpoints or artifacts.
