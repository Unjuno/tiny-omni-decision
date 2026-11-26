# AIST Bimanual Manipulation source screen — 2026-11-12

## Decision

**Status: `REVIEW_NOT_ADMITTED`.** This is a fresh metadata-only lead for a
camera + robot-state/action Decision task. It is not an approved data source
for a model run yet. No archive payload, model, train/validation labels,
sealed audit, or GPU was accessed.

## Pinned metadata and observed scale

- The [official AIST dataset page](https://aistairc.github.io/aist_bimanip_site/)
  identifies the release as AIST-Bimanual Manipulation, describes teleoperated
  bimanual robot demonstrations, places a CC BY 4.0 notice in the dataset
  header beside the downloads, and provides task archives. This is a
  **provisionally favorable source-level rights screen**. Before payload use,
  preserve the notice and verify its scope over the archive's RGB/depth media,
  state/action data, prompts, and derivatives; participant/privacy and any
  third-party-content scope remains unverified. The [official AIST release
  announcement](https://www.aist.go.jp/aist_j/press_release/pr2025/pr20250902/pr20250902.html)
  confirms the dataset is publicly downloadable and includes about 10,000
  action sequences.
- The official [dataset details](https://aistairc.github.io/aist_bimanip_site/dataset.html)
  document synchronized RGB/depth views, follower `qpos`/`qvel`, leader joint
  `action`, and text prompts/embeddings. It states 30 FPS RGB and 50 Hz joint
  data. This supports a plausible video + proprioception + optional text
  experiment. Audio is **not admitted as a modality**: the task CSV's
  environment description says “audio in hdf5,” but the official HDF5 schema
  does not list an audio field.
- The official task-list CSV was fetched as metadata only on 2026-11-12
  (28,827 bytes; SHA-256
  `62bd7daf3b43792575624cfc8942e32c65afcd0a1ff0f875850c3c666de10e33`). It
  has 119 task rows and sums to 12,025 demonstrations.
- The official `aistairc/aist_bimanip_downloader` repository was pinned at
  `43ec219724f961996ae4d65ae75edcd697650247`. Its `hdf5_links.csv` is 17,621
  bytes (SHA-256
  `e9ec61dc96f832cbc43387fd74c0ddf96527655f0b9a89111d48f415fbbb7ae5`) and
  lists 117 task archives. Those archives account for 10,705 demonstrations.
  Two task rows in the current page CSV—`fold_towel_in_half` (600) and
  `fold_towel_in_quarters` (720)—are absent from the pinned archive list. Thus
  the page's displayed 10,705 total matches the 117-archive inventory, while
  its linked current CSV sums to 12,025. The latest dataset payload revision,
  per-archive sizes, and publisher hashes are not established.

## Candidate task shape (hypothesis only)

A future Path A task could give a small Decision readout a fixed-size set of
short candidate action chunks and ask it to select the one closest to the next
teleoperated action chunk under a preregistered normalized 14-D distance and
tie/ambiguity margin. Inputs would be a synchronized RGB observation, current
robot state, and optionally the task prompt. This creates derived labels, not
native multiple-choice or success labels. Freeze candidate construction,
distance scaling, ambiguity handling, and option-ID rules before fitting.
Compare video-only, state-only, prompt-only, and joint inputs, plus missing,
cross-episode-swapped, shuffled, and time-offset controls. Do not treat another
valid demonstration as wrong, and do not call the task “both-required” unless
the matched controls show useful evidence from each stream.

The release has many nominal demonstrations, so the 200-clip floor may be
reachable if each demonstration is a distinct trajectory. The archive/task
count alone is not the statistical group count. Establish stable trajectory
IDs, and group by any shared recording session, scene, operator, or initial
configuration that can create dependence. Keep every temporal window from a
trajectory together. A task-held-out evaluation has at most 117 groups in the
pinned archive inventory; within-task held-out trajectories would answer a
narrower generalization question.

## Gates before any payload or model access

1. Preserve the official CC BY 4.0 notice and verify archive-level component
   scope for media/features and review human/third-party content. No external
   party has been contacted.
2. Pin a stable dataset revision and reconcile the 119-row task CSV with the
   117-archive download inventory; obtain official sizes and integrity hashes
   for candidate archives. Until reconciled, use only the pinned 117-archive
   intersection for planning; do not claim the 12,025 count as a stable pool.
3. Establish stable per-trajectory IDs and group independence. Confirm at
   least 200 held-out independent demonstrations after grouping by shared
   session/scene/operator/configuration; keep every window from one trajectory
   together and audit duplicate scenes.
4. Document how 30 FPS camera observations align to 50 Hz joint/action data;
   do not assume identical array indices are synchronized timestamps.
5. Freeze an action-choice target, normalized distance, ambiguity/tie rule,
   and option protocol. Before any fit, measure options-/metadata-only
   predictability and run matched video-only, proprio-only, prompt-only, joint,
   missing-stream, swapped-stream, and time-shifted controls.
6. After the preceding gates pass, check the bounded archive's disk
   requirement on D: and inspect only the selected train/validation payload to
   verify HDF5 fields, timestamps, audio presence, and human-content exposure.

Until these gates pass, do not download the archive payloads, extract features,
train, or evaluate this candidate. This screen does not change any existing
dataset, checkpoint, PR, worktree, or process.

## Live metadata-only recheck — 2026-11-12 22:04 JST

The official task CSV was fetched again in memory from the published URL. It is
still 28,827 bytes with SHA-256
`62bd7daf3b43792575624cfc8942e32c65afcd0a1ff0f875850c3c666de10e33` and 119
unique task names. `git ls-remote` confirms the downloader repository's current
`main` is still the already pinned `43ec219724f961996ae4d65ae75edcd697650247`.
The pinned `hdf5_links.csv` is still 17,621 bytes with SHA-256
`e9ec61dc96f832cbc43387fd74c0ddf96527655f0b9a89111d48f415fbbb7ae5`; joining
its ZIP filenames to task names yields exactly 117 tasks / 10,705 demonstrations.
The only task-CSV rows without a pinned archive are `0113 fold_towel_in_half`
(600) and `0115 fold_towel_in_quarters` (720). Neither metadata file exposes
per-demonstration IDs, session IDs, or publisher-provided checksums.

The official details page adds an unresolved description conflict: its feature
summary says “117 episodes,” while the current task CSV advertises 10,705 demos.
It states RGB at 30 FPS and joint data at 50 Hz, but documents no shared clock,
timestamps, or resampling map. Its sample loader indexes image and joint arrays
with the same timestep, which is not evidence that those indexes are aligned.

Provider `HEAD` requests against three official shared ZIP links (no response
body read) returned these current byte sizes on 2026-11-12 13:03 UTC:

| Task archive | Advertised demos | Bytes from provider HEAD |
|---|---:|---:|
| `spoon_yellow_block_bowl.zip` | 400 | 11,486,224,075 |
| `place_block_cup.zip` | 390 | 23,846,203,236 |
| `insert_wire_loop.zip` | 600 | 128,386,174,405 |

The provider supplied no ETag or content hash in these responses. Sizes are
planning metadata, not integrity checks. D: had 2,167,366,823,936 bytes free at
22:04 JST; C: remains full. No archive bytes, HDF5 metadata, labels, media,
features, or model outputs have been accessed.

### Gate decision

The source-level license screen is favorable: the first-party dataset page
places CC BY 4.0 beside the dataset description and download links. Keep the
payload and any derived artifacts attribution-bound; do not redistribute media.
The task/group and synchronization gates remain unresolved because public
metadata does not expose demonstration or session identities. A bounded
archive may be used next **only for HDF5 structure/timestamp/group metadata
inspection**, with no frame decoding, label-derived option construction,
feature extraction, fitting, or evaluation until a frozen grouped protocol is
possible. Hash each archive before opening it. This is the narrowest data access
that can resolve the otherwise circular group-ID gate; if the HDF5 itself has
no independent trajectory/session keys or usable time mapping, reject this
candidate without a model run.

This metadata-only inspection rule supersedes the earlier blanket “do not
download” wording above. It does not admit any archive to training or evaluation.

## Bounded ZIP inventory — 2026-11-12

Following the narrowed rule above, one archive was acquired from its pinned
official Dropbox shared link for archive-directory metadata only:

- Task: `spoon_yellow_block_bowl` (CSV count: 400 demos).
- Official shared URL:
  `https://www.dropbox.com/scl/fi/o3xtqjie51oyoz0whn21i/spoon_yellow_block_bowl.zip?rlkey=e2di9za4iy9grnp0q9tn3ef3b&dl=0`
- Download: HTTP 200, 11,486,224,075 bytes, 14m 8s elapsed (2026-11-12
  22:07:18–22:21:26 JST); measured final mean 13.59 MB/s. The first three
  30-second interval samples averaged 7.71 MB/s; later throughput was higher.
- SHA-256:
  `794c5d61724f30764218d6cadad73d2ec341558a3671c360d3580fa5d800c8de`.
- External artifact:
  `D:\CodexArtifacts\pretrained-reuse-r2\aist-bimanual-metadata-only-20261112-attempt-01\spoon_yellow_block_bowl.zip`
- The ZIP central directory contains 400 `.hdf5` members, each a separate
  `episode_N.hdf5`; numeric IDs cover 0–399 with no duplicate or missing ID.
  Their total uncompressed size is 28,321,523,304 bytes. The smallest member
  is `episode_366.hdf5` (28,407,872 bytes); the largest is `episode_65.hdf5`
  (344,472,080 bytes).

This establishes 400 stable **file-level demonstration IDs**, not 400
statistically independent sessions. The ZIP listing does not expose operator,
collection-session, scene, or initial-configuration keys. No HDF5 member was
extracted or opened; HDF5 keys, attributes, timestamps, and camera/joint clock
mapping remain UNKNOWN. No frame, sensor values, text labels, features, or model
were read; no GPU or fitting was used. Archive acquisition and ZIP directory
inspection do not admit the source to training/evaluation. The next metadata
step is to inspect one HDF5 member's key/attribute/dataset-shape metadata only;
if no reliable episode/session provenance or clock alignment can be established,
reject this candidate before reading any frame/label content.

During this work a separate C:→D: artifact-archive `robocopy /MOVE` process was
visible. It was left untouched; the new ZIP and attempt files are under a
separate D: path and were not part of that move.

## One-member HDF5 metadata gate — 2026-11-12

Using the local cached Windows `h5py 3.16.0` / NumPy `1.26.4` packages without
installing or changing an environment, only the smallest archive member was
extracted to the separate inspection directory and opened read-only:
`episode_366.hdf5` (28,407,872 bytes expanded). No dataset values were read.
The visitor recorded HDF5 object paths, kinds, shapes, dtypes, chunk/compression
metadata, and attribute names. Selected non-personal root metadata fields
(`id`, `data_name`, `creation_data`, `steps`, `camera`) were read; `author`,
`email`, task labels, text prompt contents, image bytes, and sensor/action values
were not read.

The member has 339 rows for `action`, `effort`, `qpos`, and `qvel`, each shaped
`[339, 14]` float32. Four image datasets are `[339, 20828]` uint8 encoded-byte
records, with one record per row; they were not decoded. `depth` and `sound`
are empty groups. No timestamp dataset or clock mapping is present in this
member's HDF5 structure. Equal row counts do not establish 30 FPS RGB / 50 Hz
joint synchronization. Its root `id` is 140 while its archive filename suffix
is 366; the meaning and uniqueness scope of that attribute are UNKNOWN. The
only observed creation date is `2025-07-31 16:45:38`; this is not an operator,
session, or scene identifier.

**Disposition:** stop AIST as a candidate for the current multimodal gate.
This single-member metadata result cannot establish independent session groups
or a synchronized timebase, and the archive does not supply a task split or
success labels. Do not scan/extract the remaining members, decode media, fit, or
evaluate. Retain the acquired archive and inspection file as external audit
artifacts; do not add either to Git. Reconsider only if the publisher supplies
an auditable group/time mapping and a frozen task protocol.
