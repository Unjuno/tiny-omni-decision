# Perception Test TRAIN metadata exposure — 2026-11-12

## Reconciled artifact

Added a conservative source-usage entry to
`manifests/pretrained_reuse/cohort_usage_ledger.json` for the existing
Perception Test source-candidate preflight artifact. The source revision is
`3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`.

- Records: 25,757, all tagged `train`
- Unique video groups registered: 2,149
- Existing observed validation groups overlapping this inventory: 0
- Manifest SHA-256: `655713c9f1dd61970081b28d0ebbc0f7ddb001b8ccea68576c237b8239d21430`
- Record file SHA-256: `11cd4c758b04e36702969d168a9987dde8028d75733bb95a0a684d7f712a19d6`

The inventory registers every group in the metadata file, including held and
duplicate rows, to avoid treating those source identities as unseen. Only
source/revision-scoped SHA-256 group identities are committed; raw group IDs
are not stored in the ledger. The artifact manifest states that media was not
decoded and validation/audit data was not loaded. This reconciliation read
metadata fields only and did not inspect labels, media, official test data, or
sealed audit data.

## Scope and gate

The source history remains `INCOMPLETE`. This is one additional verified
exposure record, not a complete inventory across historical worktrees and
runs. A disjoint proposed group therefore remains blocked by the fail-closed
checker, and this update does not admit a new cohort or justify training.

The stored TRAIN groups have zero overlap with the previously registered
validation groups under the same source/revision-scoped digest. That fact does
not establish semantic independence, label quality, or benchmark suitability.

## Verification

- Ledger hashes were generated with `scripts.check_cohort_usage.group_digest`.
- The focused regression asserts the exposure entry and confirms that an
  unseen group is still rejected while source history is incomplete.
- No model, checkpoint, media, sealed audit, or GPU process was changed.
