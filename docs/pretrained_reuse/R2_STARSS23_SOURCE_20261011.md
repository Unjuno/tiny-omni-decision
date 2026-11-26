# STARSS23 synchronized audio-video source probe

## Scope and source decision

STARSS23 version 1.1, immutable Zenodo record `7880637`, is a source candidate
for real synchronized environmental audio/video decisions. This is a source
and decoding probe, not model-quality evidence or admission to the legacy
Teacher corpus. No existing corpus, checkpoint or training run is modified.

The [official release](https://zenodo.org/records/7880637) supplies the media,
annotations, README and MIT license. The [authors' paper](https://arxiv.org/html/2306.09126v2)
explicitly applies MIT to the dataset and describes informed consent, ethical
review, face blurring and absence of third-party intellectual-property
restrictions. This supports ALLOW for this bounded research source probe.
The downloaded LICENSE and README are retained with SHA-256 pins in the runner.
Unofficial mirrors are not used.

## Identity and evaluation limits

- Development comprises 90 train and 78 dev-test recordings in disjoint rooms.
- Twelve train audio recordings in room 21 lack video; they cannot become AV
  pairs. Do not synthesize replacement video or count them as paired examples.
- Preserve official train/dev-test separation and group any further split by
  room and recording. Every temporal window, question and choice from one
  recording inherits the same split.
- Source IDs identify actors/sources within a scene. They are not verified
  global speaker identities, so speaker-disjoint evaluation is not established.
- There are fewer than 200 independent recordings and only 16 rooms in the
  complete development set. Room-level uncertainty estimates will be low-power.
- Simultaneous events are legitimate labels. A single-winner event taxonomy
  would discard valid information; use separately defined binary event queries
  if this source proceeds to task implementation. Merge the two speech labels
  into generic speech; gender prediction is not a proposed product task.
- The official unlabeled evaluation release and all dev-test member contents
  remain unopened. ZIP directory filenames are used only for split inventory.

## Reproducible bounded acquisition

Runner: `scripts/fetch_starss23_smoke.py`. Source metadata is retained at
`C:\CodexArtifacts\pretrained-reuse-r2\starss23-source-audit-01`.
It verifies three pinned SHA-256 values, inventories room-level splits, then
selects the first two sorted eligible dev-train recordings from distinct rooms.
No selection uses model predictions or class difficulty.

The remote audio/video ZIPs total about 7.5 GB. The runner uses exact HTTP byte
ranges with a 256 MiB transfer budget per archive, rejecting servers that ignore
Range before reading their bodies. Each selected member is limited to 128 MiB.
ZIP CRC and local SHA-256 verify individual extracted members; **whole-archive
MD5 is not verified** by partial retrieval. Upstream archive MD5 values are
recorded as provenance only.

```powershell
python scripts/fetch_starss23_smoke.py --source-dir C:\CodexArtifacts\pretrained-reuse-r2\starss23-source-audit-01 --output-dir C:\CodexArtifacts\pretrained-reuse-r2\starss23-paired-smoke-01
```

The source uses a Zenodo versioned record, not a Git revision. Do not fabricate
a 40-character Git SHA to satisfy the older dataset-manifest schema. A future
adapter needs an explicit versioned-source contract before catalog admission.

## Measured result

The process completed with PASS on 2026-10-11. Both recordings are dev-train
TAU assets, in rooms 12 and 13. Audio WAV headers report four channels at
24 kHz; first video frames decode to RGB `[960, 1920, 3]` at 29.97 fps.

| Recording | Audio duration (s) | Video duration (s) | Absolute difference (s) |
|---|---:|---:|---:|
| fold3_room12_mix001 | 253.285292 | 253.286620 | 0.001328 |
| fold3_room13_mix001 | 263.101292 | 263.063063 | 0.038229 |

Range transfers totaled 143,822,655 bytes (64,433,238 audio archive and
79,389,417 video archive), rather than downloading both full archives.
Room inventory confirmed nine train and seven dev-test rooms with no overlap.
Both splits have 78 paired recordings after excluding the twelve missing videos.

The complete small result is `R2_STARSS23_SMOKE_20261011.json`, SHA-256
`b3807ca4484bc011835dff3fcd46d9ec3330c58d0ad15b36f71efa70c3081513`.
Media and exact executed runner remain outside Git in
`C:\CodexArtifacts\pretrained-reuse-r2\starss23-paired-smoke-01`.
The source base was commit `5cafe00a58837f2220fcc9465c3d90839bdfa405` with
the new uncommitted runner; its exact SHA-256 is recorded in the result.
The committed runner has only Ruff formatting changes from that executed copy.
No model was loaded or trained. CPU suite: 254 passed, one skipped; focused
range/split tests: eight passed; Ruff and existing dataset/catalog validation
passed. The original metadata/archive files and Teacher artifacts are untouched.

## Interpretation and next gate

Follow-up: `R2_STARSS23_WINDOWS_20261011.md` records actual PCM and timestamped
video extraction for four fixed windows and 48 binary queries, including exact
save/reload checks. This remains a dev-train plumbing probe without model fitting.

Header/duration checks and a decoded frame establish media readability only.
Temporal synchronization is documented upstream; this probe does not measure
audio/video event offset or establish that vision adds predictive information.
Before fitting a Decision readout, freeze the event-query definition, grouped
splits, time windows, calibration partition and matched audio-only/video-only/
joint/shuffled-video controls. Product quality floors remain unset until the
task is defined, as the user requested. R0's 100–200M resident parameters and
<=200 MB package are provisional engineering targets, not measured attainment.
