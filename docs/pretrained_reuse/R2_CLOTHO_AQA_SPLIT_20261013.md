# Frozen development grouping for Clotho-AQA

Official train candidates from the admission audit were split by the SHA-256
of `17:<normalized uploader>` modulo100: train0–69, calibration70–84,
validation85–99. No class balancing, answer-confidence filtering or seed search.
All questions for a source remain together; inconsistent source ownership,
filename identity, duplicate normalized source/question and malformed votes fail.

| Split | Questions | Audio sources | Uploaders | Yes | No |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 2312 | 578 | 387 | 1315 | 997 |
| calibration | 550 | 138 | 96 | 321 | 229 |
| validation | 560 | 140 | 81 | 317 | 243 |

Executed pairwise intersections: zero source ID, normalized uploader and
source-plus-normalized-question overlap. Generic question wording may recur
for different audio; that is not evidence of duplicated observation content.
Waveform overlap, pretraining contamination and other-corpus overlap remain
unmeasured. These development subsets are not a new sealed audit. Official
validation/test question files remain unopened.

All split bytes and input SHA are saved in the adjacent compact report and at
`C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-development-split-01/`.
Six CPU tests passed, covering deterministic order, uploader aliases, multiple
questions per source, contradictory ownership, normalized duplicates and votes.
Ruff passed for the splitter and its tests. No model experiment or media
download has started. Next: bounded media-index validation and training-only
decode smoke before acquiring the frozen corpus and extracting features.
