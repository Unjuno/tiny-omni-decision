# CaptainCook4D release audit: metadata gates before media ingestion

Purpose: identify a real procedural-decision task with paired video, audio and
sensor evidence. This is source admission work, not a trained-model result.

Pinned official repositories:

- [Dataset](https://github.com/CaptainCook4D/CaptainCook4D/tree/1137b884e038cc128589c18eab215024018fb642)
- [Downloader](https://github.com/CaptainCook4D/downloader/tree/cf248fe2139fc6e37f9737141ec719948fbd1747)
- [Annotations](https://github.com/CaptainCook4D/annotations/tree/a8a920a3293c4db27099a20ddbe3a3a9be1283e3)

Dataset README explicitly grants Apache-2.0 for the dataset. Downloader LICENSE
is Apache-2.0; annotation repository LICENSE is MIT. Preserve both notices and
their scopes. This audit downloaded only documentation, downloader source,
download-link metadata, identity/duration metadata and split ID lists. No media,
outcome annotations, sealed examples or model predictions were loaded.

## Measured release coverage

All 384 recording IDs have download and identity metadata. Link counts:

| Stream | Recording links |
| --- | ---: |
| GoPro 360p | 383 |
| Hololens synchronized PV video | 335 |
| Raw accelerometer | 276 |
| Synchronized accelerometer | 243 |
| Microphone / audio | 0 |

The downloader README describes raw microphone support, but its actual link
manifest has no microphone, `mc`, or audio key. Thus collection capability and
downloader support do not establish released audio availability. Embedded MP4
audio has not been probed and remains UNKNOWN. Synchronized PV plus accelerometer
links coexist for 134 train, 47 validation and 62 test recording IDs in the
official person split. Links were not fetched; integrity, clock alignment,
units and real download availability remain unverified.

## Person-split metadata disagreement

Official split sizes are 204/84/96 train/validation/test, with zero repeated
recording IDs across splits. However, joining against `video_information.csv`
by recording ID gives shared person IDs:

- train/validation: 8
- train/test: 1, 2, 3, 4, 6, 8
- validation/test: 7, 8

This proves a disagreement with a person-disjoint interpretation using that
metadata; it does not prove which upstream identity source is correct. Do not
use the recording-ID prefix as a person ID. Do not silently regenerate a split
or claim subject-disjoint evaluation until the identity discrepancy is resolved.

## Decision and next gate

Do not admit this release as a verified audio/video/sensor task yet. Before any
learning, resolve identity provenance, verify a bounded training-only recording's
timestamps and streams, then freeze task and source-group splits. A possible
task is instruction-conditioned procedural-error detection. Error descriptions,
modified descriptions and correctness flags must be targets/metadata only,
never observation inputs. Missing-step records require whole-procedure evidence;
negative timestamps must not be coerced into ordinary clip intervals. Neither
multimodal necessity nor fusion improvement has been demonstrated.

Reproduction: `python -m scripts.audit_captaincook_release` writes a new directory
and refuses to overwrite. Original output:
`C:/CodexArtifacts/pretrained-reuse-r2/captaincook-release-audit-01/`.
The companion JSON pins every downloaded file's URL, SHA-256 and byte length.
No GPU work, cloud cost, existing corpus changes or prior artifact changes.
