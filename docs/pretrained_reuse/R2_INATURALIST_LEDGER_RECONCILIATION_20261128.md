# iNaturalist source-history ledger reconciliation — 2026-11-28

## Decision

**Keep the iNaturalist expansion ineligible for training and evaluation.** The
current privacy-minimized source-global ledger does not cover all identities
that the existing frozen expansion filter says were previously observed. The
remaining cohort has only 20 records, all Mallards (15 train / 5 validation),
far below the frozen 700 / 200 and eight-class requirements. No media, model,
validation score, or sealed audit was accessed in this reconciliation.

## Reproduced artifacts

The audit read the existing ledger and frozen source/filter manifests from
`C:\CodexArtifacts`; it did not contact the live iNaturalist API.

| Input | SHA-256 |
|---|---|
| Source-global ledger | `c923e97e1046d81bf068f9e3f886810f4843c29cca31a4fb6daf0634353a02eb` |
| Ascending source train | `356e081be2cb9a9d4e8789270bcd2132a37272f15ced79095e2d02eaadfc4214` |
| Ascending source validation | `7c910a7bc9b679eacdb10b5f39af7e1b7adea9d2b8780721e5c882c715a80f54` |
| Filtered train | `af631c52118c7e260ab3209185e3256238bb9d69f580c2dbaac1cc08c0afa953` |
| Filtered validation | `88ec51a0a0d57c0bc1b1a7ced28524c57f4ee441c9ab1a011239008108558e4d` |
| Filter metadata | `2c9b7cc98f65e5aed8c5e8346b684e6ebc074521ea79cb92708ef0056606b197` |

The source cohort was frozen on 2026-10-30 from a live API response. It has
1,813 rows: 1,276 train and 537 validation. The frozen filter retained 20 and
records 1,793 exclusions because those observers were present in previous
iNaturalist cohorts.

## Integrity finding

The ledger declares `history_status: INCOMPLETE` and contains 2,588 unique
observer identity digests. The filtered metadata records 2,710 previous
observer groups, a count difference of 122. More directly, matching the 1,793
excluded source rows against the ledger's canonical observer digests found:

- 1,677 excluded rows whose observer digest is present in the ledger;
- **116 excluded rows across 51 distinct observer groups whose digest is
  absent from the ledger**;
- all 20 retained rows have no observer digest match in the ledger;
- retained train/validation observer overlap is zero.

The 122 count difference is an aggregate discrepancy, not a claim that exactly
122 identities are missing. The confirmed 51-group mismatch shows that the
ledger does not fully explain the prior-cohort exclusion that produced this
filtered split. The audit does not infer whether those absent digests are
missing historical model exposures or conservative metadata-only exclusions.
Until their provenance is reconciled, the global history cannot be certified
complete and no later mutable API query can be called fresh from count data
alone.

This is a data-provenance limitation, not evidence about model capacity. It
also explains why the live multi-taxon paired count screen (4,426 Aves,
484 Mammalia, 228 Amphibia before historical filtering) cannot be promoted to
a training corpus: counts do not establish unseen identities, distinct events,
or photo/audio correspondence.

## Implementation and verification

Added `scripts/audit_inaturalist_exposure_ledger.py`. It is an offline,
read-only reconciler; it emits file hashes and aggregate counts, never raw
observation, photo, or observer IDs. It exits nonzero while history is
unreconciled. The machine result is stored outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2-local\inat-ledger-reconciliation-20261128\summary.json`.

Focused tests:

```text
py -3 -m pytest tests/test_inaturalist_exposure_ledger.py -q
4 passed
py -3 -m ruff check scripts/audit_inaturalist_exposure_ledger.py tests/test_inaturalist_exposure_ledger.py
All checks passed
```

The real-artifact audit produced exit code 2 as designed because history is
not reconciled. No iNaturalist source/media, legacy artifact, pre-existing
worktree, or running process was changed. No GPU or cloud compute was used.

## Next data gate

Do not expand this iNaturalist task or train on the 20-row remainder. A future
cohort needs a frozen source revision, sufficient genuinely new independent
groups after a complete exposure-ledger reconciliation, all required classes,
and a task where the paired modalities concern the same event. Resolve the
source-global history gap before interpreting additional live API counts.
