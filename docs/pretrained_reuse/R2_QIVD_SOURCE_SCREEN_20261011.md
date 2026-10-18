# Qualcomm Interactive Video Dataset source screen — 2026-10-11

## Decision

Keep QIVD **review-only**. Do not access the download workspace, accept the
data agreement, fetch the annotations/videos, add a dataset manifest/catalog
entry, or run a model on it.

QIVD is relevant in form: the official page describes 2,900 videos with audio,
spoken user questions, human transcripts and answers, and answer timestamps.
It lists 13 question categories, but its category table shows only 22 examples
in the `Audio-Visual` category. The license is not a permissive open-data
license: download/use accepts Qualcomm Technologies' Data License Agreement
for internal research use, which defines Research Use as non-profit and
excludes all Commercial Use, including use of Results in a product or service.
Its restrictions also prohibit combining the licensed data with third-party
data and impose confidentiality/data-protection duties. This is not compatible
with using QIVD to select or qualify a product model without further rights
review. The narrow 22-example Audio-Visual category also falls far below the
roadmap's provisional 200 independent-scene floor.

The candidate has no immutable data revision or public source-identity/split
manifest established in this screen. The downloads page requires workspace
access and says to download all files. None of those files were accessed. No
dataset terms were accepted; no QIVD examples, model inputs, or annotations
were loaded.

## Pinned observations

| Item | Observation |
|---|---|
| Official project page | Qualcomm's QIVD overview; retrieved 2026-10-11. It describes 2,900 video/audio items, 13 categories, questions and answers, and answer timestamps. |
| Category counts | The official table lists 22 samples in the Audio-Visual category and 2,878 across the other 12 categories. The small AV subset is not enough for the roadmap's provisional independent-scene evaluation floor. |
| Media | The project says the videos are crowdsourced. The page provides `videos.zip` and `annotations.zip` through its developer download workspace. No file size, source-level rights, contributor-consent record, immutable dataset version, or source-group split was verified. |
| Agreement | Qualcomm Technologies, Inc., “Data License Agreement – Research Use,” dated 2025-03-24. Download, installation, or use constitutes acceptance. The agreement grants only limited, non-transferable, non-sublicensable internal processing for Research Use; it defines Research Use as solely non-profit and excludes Commercial Use. |
| Restrictions | The agreement prohibits third-party data combination and third-party disclosure; it includes confidentiality, privacy, de-identification, retention, termination and deletion obligations. The official page also says the dataset is intended for research. |
| Project fit | Its 22 Audio-Visual-category cases cannot support a paired AV confidence interval or the 200-scene provisional floor. The broader dataset includes many video-centric categories and should not be presumed to require both audio and video. |
| Status | No data access, media fetch, annotation inspection, processing, training, evaluation, or catalog admission. `trust_status` remains REVIEW. |

This screen does not determine whether a non-commercial research-only study
would be lawful or useful. It establishes that QIVD cannot currently serve as
the product-aligned joint-AV quality gate in R2. A future research-only use
would need an explicit user decision after reviewing the agreement's scope and
operational obligations, and would still need an adequately sized AV task.

## Sources

- Qualcomm, [QIVD overview, data format, category counts and license notice](https://www.qualcomm.com/developer/software/qualcomm-interactive-video-dataset-qivd), retrieved 2026-10-11.
- Qualcomm, [QIVD downloads page](https://www.qualcomm.com/developer/software/qualcomm-interactive-video-dataset-qivd/downloads), retrieved 2026-10-11. It links the data through Qualcomm's developer workspace; no file was fetched.
- Qualcomm Technologies, Inc., [Data License Agreement — Research Use, dated March 24, 2025](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/qvid-dataset-research-license-mar-27-2025.pdf). Relevant clauses: definition of Research Use and Commercial Use (§1), limited research license (§2), restrictions on third-party data and commercial use (§3), confidentiality (§6), and personal-data handling/retention (§7).
