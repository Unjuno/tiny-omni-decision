# EgoViz-120 session-group split preflight — 2026-10-29

## Result

The existing 870,961-byte metadata index was re-read and hash-verified
(`index.parquet` SHA-256
`c83126637cc24a7ac2b3600724b50cf9c53966b9b098acc57d5534e14ade1074`). A
deterministic 80/20 metadata-only split was formed by sorting
`group_id` values by `SHA256("seed17:egoviz-session-split-v1:" + group_id)`;
144 session groups / 4,335 clips fell in train and 36 groups / 1,057 clips in
development. There was no group overlap, and no `sequence_id` crossed session
groups. All 12 publisher categories appeared in both sides.

| Category | Train clips / groups | Development clips / groups |
| --- | ---: | ---: |
| cleaning_kitchen | 131 / 22 | 5 / 3 |
| cleaning_restroom | 242 / 43 | 90 / 11 |
| cleaning_windows | 143 / 29 | 29 / 8 |
| cooking_food | 136 / 8 | 57 / 2 |
| managing_laundry | 179 / 11 | 31 / 2 |
| managing_trash | 160 / 54 | 48 / 15 |
| mopping_floor | 1,300 / 112 | 295 / 28 |
| organizing_items | 226 / 83 | 53 / 19 |
| preparing_cleaning_supplies | 674 / 111 | 116 / 21 |
| preparing_food | 350 / 20 | 191 / 5 |
| sweeping_floor | 128 / 45 | 31 / 13 |
| wiping_surfaces | 666 / 91 | 111 / 24 |

The exact hashed group assignments and counts are in the external report
`C:/CodexArtifacts/pretrained-reuse-r2/egoviz-session-split-preflight-20261029-attempt-01/report.json`
(SHA-256
`d2e2869dcce0a7d96cddc7dee639794a7cf253c3f922637ab9c9699cb5e6d7d2`).
The repository manifest pins that receipt.

## Admission decision

This establishes only that a deterministic **session**-disjoint metadata split
can include all categories. It does not establish participant independence:
the publisher has not verified that separate `group_id`s belong to separate
people. The split is not an official split and is not a locked audit. The
publisher says the labels are model-generated and not human-verified; two
development categories (`cooking_food`, `managing_laundry`) have only two
session groups each, and `cleaning_kitchen` has only three. No model, score,
raw media, sensor payload, or sealed audit was loaded.

The source remains **NOT ADMITTED** for training or quality claims. This index
does not establish that an action decision requires both video and IMU, and
there is no audio stream. The smallest indexed MCAP is 150,184,348 bytes, the
median is 668,009,041 bytes, and the full payload is about 6.8 TB; local C:
space was about 0.72 GiB. No payload was downloaded. Before considering even a
bounded media prototype, verify participant grouping and action-label quality,
confirm raw timestamp alignment from one MCAP, and secure enough local storage
for download, extraction, and replay. Do not use the dashboard MP4 overlays as
model input.

