# CaptainCook identity history follow-up

Checked upstream file history to test whether the person-split mismatch was
introduced by the 2024 metadata rename. It was not: all 384 surviving recording
IDs have identical person IDs in historical `video_info.csv` at
`842fb501b607ebe48494fd93fa0d933a9a22108d` and the pinned current
`video_information.csv`. The older table has 388 records; the four removed IDs
are 16_110, 17_2, 26_37 and 4_120. No outcome annotations were fetched.

| Official split | Person counts from current identity table |
| --- | --- |
| Train | 1:3, 2:1, 3:55, 4:55, 5:43, 6:46, 8:1 |
| Validation | 7:43, 8:41 |
| Test | 1:52, 2:34, 3:1, 4:1, 6:3, 7:3, 8:2 |

The split file's latest relevant commit is
`7b3e7fdda035b416d13c25f435dada3ee9cdf6e2` (2023-12-16), while the table
rename is `0e9a108be2cbcbcbd592e7418c0ab9c16232d27a` (2024-05-07).
This history does not identify which assignments are wrong. Majority membership
is not ground truth. Do not repair by silently relabeling people, moving selected
examples, or dropping inconvenient evaluation records.

The published benchmark remains a possible recording-disjoint reference, but
cannot currently support a verified person-disjoint claim. Keep the source's
person-generalization learning gate closed. A task-specific source group policy
requires authoritative identity resolution or a separately justified grouping
before labels and outcomes influence selection. Audio existence is independently
established for one training MP4; cross-device sensor alignment is still unknown.

Artifacts:
`C:/CodexArtifacts/pretrained-reuse-r2/captaincook-identity-history-01/`.
The companion JSON records the historical URL/hash and exact counts. No media
download, GPU work, learning, sealed-set access or legacy-artifact changes in
this follow-up. This bounds the historical-file hypothesis; repeating that
lookup cannot resolve identity truth.
