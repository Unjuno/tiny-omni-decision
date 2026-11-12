# Perception Test source-exposure recheck (2026-11-12)

## Finding

The current ledger already covers every official TRAIN video group listed in the
preserved source-candidate metadata artifact. The artifact contains 25,757 rows
and 2,149 unique `group_id` values. Recomputing the source/revision-scoped group
hashes produced 2,149/2,149 overlaps with the ledger's known exposure hashes.
The existing cohort checker returns `BLOCKED_PRIOR_GROUP_USE` for that complete
TRAIN group set. The existing 1,518 development-validation video groups are
also registered as already observed.

Therefore, no fresh official-TRAIN group remains in this candidate inventory,
and the already observed validation set cannot be reused for a new blind quality
claim. Do not use the official test split or any sealed audit as a substitute.
Keep the source history `INCOMPLETE` and continue to fail closed: this recheck
covers the identified source artifacts, but does not prove exhaustive historical
exposure across every worktree, run, or external cache.

## Evidence

| Artifact | SHA-256 | Rechecked fact |
|---|---|---|
| `manifests/pretrained_reuse/cohort_usage_ledger.json` | `05df0b6e6518e79db6479c655f3e9ca67bc4f4f8208a7b22392182b664bcd43b` | `INCOMPLETE`; 1,518 observed validation and 2,149 train metadata-exposure groups |
| Source-candidate `manifest.json` | `655713c9f1dd61970081b28d0ebbc0f7ddb001b8ccea68576c237b8239d21430` | Pinned source revision `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`; 25,757 rows; metadata-only preflight |
| Source-candidate `source_candidate_records.jsonl` | `11cd4c758b04e36702969d168a9987dde8028d75733bb95a0a684d7f712a19d6` | 65,902,665 bytes; 2,149 unique source video groups |

SHA-256 of the sorted 2,149 source/revision-scoped group hashes joined with LF:
`d34a9ec8deb3229a24b3d6fb9a5bbf772f7d3639b6d6ee71aaf89c59993c3f99`.

The recheck used the existing `check_cohort` implementation directly against
those group IDs and observed `BLOCKED_PRIOR_GROUP_USE`, `overlap_count=2149`,
and `history_status=INCOMPLETE`. Existing receipts also record train-side
feature/readout and runtime work plus one public development-validation
readout; these do not create unused groups. The existing validation prediction
and report hashes remain registered in the ledger.

## Handling and disposition

This was a metadata exposure audit. It read the existing derived source-candidate
records only to count `group_id` values and compare hashes; other row fields
were not used or reported. No upstream source annotation archive, media payload,
official test split, or sealed audit was opened. No model was loaded, no GPU or
training was used, and no legacy artifact was changed.

Perception Test is not eligible for another quality evaluation on its official
TRAIN or already observed validation groups. The full history remains
`INCOMPLETE`; no claim of complete source-wide history is made. Select a different
source only after its license, independent-group, split, and task-fit gates pass.
