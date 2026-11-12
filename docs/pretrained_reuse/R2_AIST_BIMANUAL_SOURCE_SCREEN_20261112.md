# AIST Bimanual Manipulation source screen — 2026-11-12

## Decision

**Status: `REVIEW_NOT_ADMITTED`.** This is a fresh metadata-only lead for a
camera + robot-state/action Decision task. It is not an approved data source
for a model run yet. No archive payload, model, train/validation labels,
sealed audit, or GPU was accessed.

## Pinned metadata and observed scale

- The [official AIST dataset page](https://aistairc.github.io/aist_bimanip_site/)
  identifies the release as AIST-Bimanual Manipulation, describes teleoperated
  bimanual robot demonstrations, displays a CC BY 4.0 notice, and provides
  task archives. The [official AIST release announcement](https://www.aist.go.jp/aist_j/press_release/pr2025/pr20250902/pr20250902.html)
  confirms the dataset is publicly downloadable and includes about 10,000
  action sequences. The exact applicability of the webpage license to every
  media/derived component and participant privacy scope still needs review.
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

A future Path A task could ask a small Decision readout to select the recorded
next action chunk from supplied alternatives, conditioned on a synchronized
camera observation, robot state, and task prompt. It would compare video-only,
state-only, prompt-only, and joint inputs, then test missing, shuffled, and
temporally offset controls. This is action prediction from demonstrations,
not a success/failure or task-completion benchmark: the source pages do not
document success labels. The option construction must avoid treating other
valid demonstrations as incorrect actions.

The release has many demonstrations, but independence is not settled. Grouping
by task archive yields only 117 source groups, below the roadmap's provisional
200-scene/clip floor. Episode-level grouping might provide enough held-out
clips for within-task generalization, but this must be justified from the
actual file/episode identity scheme and checked for repeated or near-duplicate
scenes before any fit. A task-held-out evaluation has at most 117 task groups.

## Gates before any payload or model access

1. Establish that the CC BY 4.0 grant on the official page applies to the
   downloadable media and derived features, and review any human/third-party
   content obligations. No external party has been contacted.
2. Pin a stable dataset revision and reconcile the 119-row task CSV with the
   117-archive download inventory; obtain official sizes and integrity hashes
   for the candidate archives.
3. Verify the intended episode grouping, task/prompt overlap policy, and
   independent held-out sample count. Keep all episodes from one recording
   together and audit duplicate scene content.
4. Freeze an unambiguous action-choice target and hard-negative policy. Do not
   label another valid demonstration as wrong; establish which video and state
   inputs are needed using preregistered unimodal and perturbation controls.
5. Check the bounded archive's disk requirement on D: and confirm the exact
   HDF5 fields, timestamps, audio presence, and human-content exposure only
   after the preceding gates pass.

Until these gates pass, do not download the archive payloads, extract features,
train, or evaluate this candidate. This screen does not change any existing
dataset, checkpoint, PR, worktree, or process.
