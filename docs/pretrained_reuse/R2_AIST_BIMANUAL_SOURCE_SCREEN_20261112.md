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
