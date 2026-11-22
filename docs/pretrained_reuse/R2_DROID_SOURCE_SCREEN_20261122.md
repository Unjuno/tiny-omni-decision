# R2 DROID Source Screen — 2026-11-22

## Decision

**Do not admit DROID `droid_100` as a train/validation quality source for the current Path A Decision work.** It is a parser/runtime smoke source only. No model was trained or scored on DROID. The one inspected episode is quarantined from all later training, validation, and evaluation.

This is a source-screen decision, not a claim that the full DROID release is unusable. The official docs describe `droid_100` as a 100-episode debugging sample and the full RLDS release as approximately 1.7 TB. The source records are robot demonstrations with continuous actions and language instructions; the debug sample does not provide an independent validation split. The official schema describes raw per-episode metadata such as building and data-collector IDs, but the inspected RLDS example itself exposes only `file_path` and `recording_folderpath` as episode metadata. The queried raw prefix for this example did not provide the corresponding trajectory or metadata JSON, so a defensible scene/contributor group key could not be recovered.

## Integrity finding

The first record of the first `droid_100` train shard was parsed only to check the TFRecord schema. Its `episode_metadata/file_path` contains a `/success/` directory component. This is an outcome label embedded in the path and would leak the answer if the path were available to a model or used to form a target. The path was not used as model input, and the record was not used for fitting or evaluation. Future use is prohibited for this record; its serialized-record and path-derived identity hashes are recorded in the manifest without retaining the source path.

The debug subset has only a `train` split, 100 episodes across 31 shards, and no source-published held-out split. Hashing `recording_folderpath` would distinguish episodes, but would not establish that training and evaluation scenes, buildings, or collectors are disjoint. Randomly splitting these episodes would therefore not satisfy the project’s group-integrity requirement.

## Reproduction and pinned evidence

- Official documentation: [The DROID Dataset](https://droid-dataset.github.io/droid/the-droid-dataset)
- Official dataset code repository: [droid-dataset/droid](https://github.com/droid-dataset/droid)
- Official docs identify `droid_100` as a 2 GB debugging sample, full RLDS as about 1.7 TB, and raw metadata as the source for building/data-collector identity.
- Local source-screen receipt and file SHA-256 values: [`droid-source-screen-20261122.json`](../../manifests/pretrained_reuse/droid-source-screen-20261122.json)
- Data bytes remain outside Git under `C:\CodexLocalArtifacts\pretrained-reuse-r2\`; no media or dataset shard was added to the repository.
- The existing local TensorFlow/VideoPrism environment was used without dependency upgrades. The smoke parser ran on CPU. It read the first serialized record only.

## Consequence

Do not download the remaining debug shards or the full RLDS release for this task. Reconsider DROID only if an official, pinned episode-to-scene/building/contributor mapping and a non-leaking, group-disjoint task/split can be obtained at a practical scale. Until then, keep searching within the existing R2 source-admission gates. No sealed audit was accessed.
