# AVA Active Speaker label and split audit — 2026-10-22

## Scope and result

This is an additive follow-up to the 2026-10-18 source screen. It reads only the
previously present official train and validation annotation archives under
`C:\CodexArtifacts\pretrained-reuse-r2\ava-activespeaker-label-screen-01`.
No video/audio media was opened, fetched, decoded, or embedded. No model was run,
no checkpoint was selected, and no label metric was calculated. The official
challenge/test archive was not used for any task, split, candidate, or result.

The train and validation archives use one CSV per video and provide 8 columns:
video ID, timestamp, box coordinates, speaking/audibility label, and entity
track ID. Grouping on video IDs gives zero train/validation video overlap. This
does not establish participant independence, near-duplicate independence, or
underlying media rights.

| Split | Archive SHA-256 | Videos | Annotation rows | Unique video/entity tracks | `SPEAKING_AND_AUDIBLE` | `NOT_SPEAKING` | `SPEAKING_BUT_NOT_AUDIBLE` |
|---|---|---:|---:|---:|---:|---:|---:|
| Train | `528594517178b2d04ab28914d35c95c3e2bdca5c46e7af359921968bef60c3d5` | 120 | 2,676,314 | 29,723 | 682,404 | 1,969,134 | 24,776 |
| Validation | `ced431ed2006deacd3968954d309c5b780179866b0f63d5d17d4df8a8ec81622` | 33 | 768,307 | 8,015 | 192,748 | 567,815 | 7,744 |

Train archive size is 18,187,972 bytes; validation is 5,097,948 bytes. Both
hashes are from the preserved local archives. The label totals above are a
source-structure audit, not model performance.

## Integrity incident and containment

While checking local archive inventory, a wildcard archive-listing command
also opened an additional test-named archive and enumerated its archive member
metadata (paths and member sizes). It did **not** extract or read any member
payload, CSV row, annotation, label, or media, and no test-derived statistic,
ID, or result was used for this report, model selection, or any decision. The
test-named archive is excluded from all future work in this goal. The existing
local file was not changed. Because even its member metadata was inadvertently
inspected, this incident is disclosed and this test archive must not be
represented as an untouched blind audit for this work.

## Admission decision

**REVIEW ONLY; no fitting or quality evaluation admitted.** The official Google
AVA page describes the released dataset as CC BY 4.0, but the included active-
speaker videos are YouTube-origin media and source-level rights for their use
remain unresolved in this project. The official validation split has only 33
video groups, below the project's proposed 200-independent-group minimum for a
quality gate. Millions of frame/face rows do not increase the number of
independent videos. Speaker/person grouping beyond within-video `entity_id`,
video availability, and usable synchronized media are also unverified.

If the rights and media questions are resolved later, this could support a
low-power development diagnostic for supplied-face-track AV correspondence,
with all rows grouped by video and matched audio-only, video-only, and joint
controls. It cannot close the general AV quality gate or support a 95%/98%
claim. Do not use this exposed validation for broad checkpoint or hyperparameter
selection.

## Prior source

The official source page states that the AVA datasets listed there are released
under CC BY 4.0 and documents Active Speaker row labels and video partitions:
[Google Research AVA downloads](https://sites.research.google/gr/ava/download/).
The earlier source screen remains at
`R2_AVA_ACTIVE_SPEAKER_SOURCE_SCREEN_20261018.md`; this report supersedes only
its statement that local train/validation annotation archives had not yet been
inspected. It does not alter the earlier rights limitation.
