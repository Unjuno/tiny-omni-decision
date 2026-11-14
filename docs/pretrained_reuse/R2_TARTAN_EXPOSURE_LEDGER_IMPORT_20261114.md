# Tartan metadata exposure added to the cohort ledger — 2026-11-14

## Decision

Register the known Tartan Aviation parent-clip identities as prior metadata
exposure. Keep all three source revisions at `history_status: INCOMPLETE`.
The source is not admitted for model fitting or quality evaluation: this
registration records only known exposure and does not establish dataset or
underlying-media rights, complete cross-worktree history, or an eligible task.

## Evidence

- External roster:
  `D:\CodexArtifacts\pretrained-reuse-r2\tartanaviation-atc-adbscreen-20261114-attempt-01\group_exposure_roster-v2.json`
- Roster SHA-256:
  `6426469A01127947190F570ED0776A77CBE61D73C7B59E44520500BEAFA4C9F3`
- Metadata screen verified all 184 Parquet shard hashes and read only the
  `utterance_id` column; no media, transcript, or callsign content was read.
- The roster contains 40,899 distinct parent-clip identity hashes for each of:
  - `castacks/TartanAviation@077a04abac1e50953c45b72524edd79e3a21d9bd`
  - `twangodev/tartanaviation-atc-adsb-utterances@bc588362d2f877e6ba0910fe7b6e035649d43918`
  - `twangodev/tartanaviation-atc-labels@0775355c388aae64335522f40bff9fc5129b2922`
- The ledger stores only source-scoped SHA-256 group identities, not raw clip
  IDs. The metadata exposure role is explicitly not a model-fit role.
- Existing broader exposure history and speaker/aircraft-level identity
  grouping remain unknown. Any proposed use therefore remains blocked.

## Validation

The importer rejects unsupported roster shape, malformed hashes, duplicate
source revisions, and any history-changing or non-metadata exposure claim. It
preserves the original ledger and records the import evidence hash. Focused
tests confirm that a known group is blocked as prior use and a new group from
an `INCOMPLETE` source remains blocked. This is an exposure-accounting update,
not an admission decision or model-quality result.
