# VisTouch synchronized audio-video-touch split audit — 2026-10-26

## Relationship to the prior screen

This is a follow-up to the pinned metadata-only screen already saved on
2026-10-14 at
`C:\CodexArtifacts\pretrained-reuse-r2\vistouch-source-screen-01\receipt.json`
(SHA-256 `85B86F9B50E8067408DB5E70EF326F92E238C41932A004FEE12C7F0A3D6DCE7E`).
That receipt already pins the same source commit and metadata. This turn
revalidated those hashes and adds the exact split/session/force overlap audit
below; it does not present VisTouch as a newly discovered source. The prior
screen had not accessed media, and this follow-up did not either.

## Disposition

**Promising for a low-power, task-specific Path A mechanics probe; not yet
admitted for feature extraction, fitting, checkpoint selection, or product
quality claims.** VisTouch provides synchronized contact audio, video, and
force-tactile time series for an 8-way material-recognition task. Its metadata
offers a useful cross-force split, but there are only 24 continuous sessions
(eight material classes × three force conditions), and the official 9N test
has already been used for a published baseline. The test must not be used to
select a new model. This does not close the four required text/image/audio/video
task battery or the natural AV M-2 gate.

## Pinned source and rights

- Official project repository: [liangnjupt/VisTouch](https://github.com/liangnjupt/VisTouch), default branch `main`, pinned at commit `a569c7e572a78f232e819b652e0ebb718952e476` (2026-08-19).
- Pinned Git tree: `3d752cff8c4b54c8d78af0b4648f9749109a9036`.
- The repository's [`DATA_LICENSE`](https://github.com/liangnjupt/VisTouch/blob/a569c7e572a78f232e819b652e0ebb718952e476/DATA_LICENSE) expressly applies CC BY 4.0 to files under `audio/`, `tactile/`, `video/`, and `metadata/`; scripts are separately MIT. The GitHub API's repository-level license field is `NOASSERTION`, so this disposition relies on the explicit pinned data-license file, not the repository classifier.
- Upstream describes a fixed camera, contact microphone and dexterous-hand force sensor. The released video is 640×480 at 30 fps; audio is 16 kHz; tactile is resampled to a 100 Hz grid.
- The data bundle itself is hosted on a Baidu share, not in the Git repository. The pinned README gives a public extraction code. The share page was reached for a metadata-only check, but the browser safety gate refused to open it for further interaction. No archive, media, or author-provided feature payload was downloaded. Do not bypass that site gate; proceed only if data becomes available through a permitted source.

## Metadata and split audit

The pinned GitHub tree contains 10,498 clip-index rows, 312 source-sample rows,
2,000 segment rows, and 24 session rows. The clips are not independent
observations: each 2-second clip is derived from a 0.5-second source window at
0.15-second stride, so neighboring clips overlap by 70% of the source window.
All comparisons must group by source session/sample, never by clip row.

The metadata audit found:

| Split | Clips | Source sessions | Force | Material classes |
| --- | ---: | ---: | --- | ---: |
| train | 7,019 | 16 | 3N and 6N | 8 |
| test | 3,479 | 8 | 9N | 8 |

There are zero session IDs, sample IDs, or overlapping source-time intervals
shared across the upstream train and test splits. The 9N test comprises one
session per material and is a force-condition generalization test for these
same eight materials; it is not evidence of generalization to unseen material
classes or populations. Upstream reports its random-forest handcrafted-feature
baseline on a filtered `half` subset as audio 51.6%, video 51.2%, tactile 19.4%,
and all-three fusion 65.4% on 2,506 examples. These are upstream values, not
reproduced here and not directly comparable to all 3,479 metadata test rows.
Because this result has been published, the official test is considered
observed and closed to iterative selection.

A possible development-only mechanics comparison, if media access is resolved,
is to train on complete 3N sessions and validate on complete 6N sessions,
with all 8 classes on each side. This yields only eight independent validation
sessions and reuses the same material classes. Treat it as low-power, scoped
cross-force evidence; it cannot meet R0's 200-independent-group quality gate.
Compare audio-only, video-only, tactile-only, and multimodal heads on identical
clips/options, but do not presume that the task requires all inputs. Keep
calibration metrics and per-material results, and report session-grouped
uncertainty.

## Data-quality and task caveats

- Upstream's quality report says the released micro-clips are deterministic
  interpolations of real source windows; they are not additional physical
  captures. The source segments tile 24 continuous sessions without gaps over
  20 ms and without overlap, but windows inside each segment intentionally
  overlap.
- Audio and tactile use wall-clock timestamps; video has no independent
  timestamp file and is aligned linearly by frame index. Upstream describes
  this as approximate, not guaranteed frame-accurate. This is suitable only
  for the coarse segment task until independently verified.
- Force is tied to split: training uses 3N/6N and test uses 9N. It is a
  cross-pressure condition, not an IID random split. The test has one session
  per material, so session-bootstrap uncertainty is extremely coarse.
- The task is material recognition from contact signals, not a broad world
  reasoning task. Audio/video/tactile complementarity must be measured with
  matched unimodal controls; synchronization and the upstream fused result do
  not prove that every modality is needed.

## Local audit artifacts and hashes

Metadata-only source files were saved outside Git under
`C:\CodexArtifacts\pretrained-reuse-r2\vistouch-source-audit-a569c7e-attempt-01\`.
SHA-256:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `classes.json` | 1,020 | `025D7303623A112666D3D0FAB55148DD8BFB8348FC76C2FFDBD4EF3DF76549AD` |
| `clips_index.csv` | 2,977,824 | `48B2EFCE42E1E95EDD81ACA181C6988B9C7B885CC2E25D9432BF48439594BF75` |
| `samples.csv` | 106,858 | `A4CFD0C488FDBBB17A461C7888E4D66364A2E2ECF764595D5F4D1BD42CC54604` |
| `segments.csv` | 216,123 | `38755930872D9F434E5E38BEA02D0731148465591D852924112DF5774775B981` |
| `sessions.csv` | 5,640 | `72A9254592DBCD11DD32F0B485E55C3D3CDE0E6E8B8DCB33141076E2D9834213` |

These hashes pin the inspected metadata copies. The audit did not verify
media-file hashes, model-feature contamination, or underlying specimen
independence. No GPU, training, or sealed-data access occurred.
The machine-readable row/split summary is
`C:\CodexArtifacts\pretrained-reuse-r2\vistouch-source-audit-a569c7e-attempt-01\metadata-audit-summary.json`,
SHA-256 `99D7E460B32C1898632D9A2209F98D82D97115DB3F7EC4FFD8AE972A1FE054B7`.

## Sources

- [Pinned VisTouch README and dataset description](https://github.com/liangnjupt/VisTouch/blob/a569c7e572a78f232e819b652e0ebb718952e476/README.md)
- [Pinned explicit data license](https://github.com/liangnjupt/VisTouch/blob/a569c7e572a78f232e819b652e0ebb718952e476/DATA_LICENSE)
- [Pinned metadata tree](https://github.com/liangnjupt/VisTouch/tree/a569c7e572a78f232e819b652e0ebb718952e476/metadata)
- [Upstream material-recognition report](https://github.com/liangnjupt/VisTouch/blob/a569c7e572a78f232e819b652e0ebb718952e476/docs/classification_report.md)
- [Upstream data-quality report](https://github.com/liangnjupt/VisTouch/blob/a569c7e572a78f232e819b652e0ebb718952e476/docs/data_quality_report.md)
- [Upstream synchronization/alignment report](https://github.com/liangnjupt/VisTouch/blob/a569c7e572a78f232e819b652e0ebb718952e476/docs/alignment_report.md)

## 2026-10-26 acquisition recheck

The user authorized Computer Control to obtain the public media. The browser site-safety control blocked opening the official Baidu share, and explicitly disallowed retrying through another browser surface or an indirect download path. The pinned upstream repository still provides only that share; its GitHub Releases page contains no release assets, and no publisher-provided direct archive URL was found in the official repository. No archive listing, credential flow, or media download was accessed. This source remains REVIEW for media access and is not admitted to fitting. Proceed only if the publisher supplies a direct public archive URL or the user places the archive in the work environment. The metadata-only screen and previous artifacts remain unchanged.
