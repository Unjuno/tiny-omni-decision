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

The follow-up inventory pass scanned 60 JSONL files across 35 direct
`inat-*` artifact directories under `C:\CodexArtifacts` without following
links or writing raw IDs. The canonical file-manifest SHA-256 is
`91bda6749cbb1a6e60821f6a6beeba786a58da7c606ee0cc6d26982599dfc11b`.

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

Across the broader local inventory, 40,990 JSONL row occurrences contained
2,816 unique hashed observer groups. The old ledger covered 2,588 and lacked
228 of these. This is a lower-bound artifact inventory: rows repeat between
source manifests, feature receipts, and predictions, and the scan does not
prove which unlisted groups reached a model. It also cannot certify unscanned
worktrees or artifacts outside the `inat-*` directories. The 228 groups must
not be labeled model-exposed without provenance; they do show that the old
ledger is not a complete inventory of locally retained iNaturalist identities.

The 122 count difference is an aggregate discrepancy, not a claim that exactly
122 identities are missing. The confirmed 51-group mismatch shows that the
ledger does not fully explain the prior-cohort exclusion that produced this
filtered split. The audit does not infer whether those absent digests are
missing historical model exposures or conservative metadata-only exclusions.
Until their provenance is reconciled, the global history cannot be certified
complete and no later mutable API query can be called fresh from count data
alone.

## Artifact-level provenance follow-up

The inventory tool now emits per-JSONL aggregates (row count, unique observer
groups, and groups absent from the ledger), without emitting observer or media
IDs. This narrows where the discrepancy appears while preserving the privacy
boundary. The follow-up output is stored outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2-local\inat-ledger-reconciliation-20261128-attempt-03\summary.json`
(SHA-256 `d3c75fdf8219f337e853252eaca97d4c522af14957700ebad50f2a7605289f23`).

- All 20 rows in the retained candidate split (15 train / 5 validation) have
  observer groups absent from the old ledger. This independently confirms that
  the retained set cannot be certified as fresh against the incomplete ledger.
- The currently recorded `freshval-v2-merged-features` training input contains
  1,134 unique observer groups, all present in the ledger. The older base
  VideoPrism train/validation feature caches also have no groups absent from
  that ledger. Thus this inventory did not find a missing-ledger group in those
  identified model feature inputs.
- An intermediate incremental feature cache contains four groups absent from
  the ledger, but the selected readout's recorded `feature_dir` points to the
  merged-v2 cache above. We therefore do not infer that those four groups were
  consumed by the selected readout.
- Other unledgered groups occur in frozen source and validation manifests,
  including cohorts that are not the selected readout's recorded feature
  input. The broad count of 228 still cannot be equated with model exposure.

This follow-up resolves one narrow ambiguity about the recorded readout's
feature cache; it does not repair the source-global ledger, certify all other
historical exposure, or make the 20-row remainder useful training data. No
additional model, validation, media, or sealed-audit evaluation was run.

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
`C:\CodexArtifacts\pretrained-reuse-r2-local\inat-ledger-reconciliation-20261128-attempt-02\summary.json`
(SHA-256 `dcf152c2a693c85e5eef8b7cd21a6d24f598975e96c062b0b0809e803afb2d2b`).
The CLI compares the fixed cohort alone or accepts `--inventory-root` to
compare hashed observers from local iNaturalist JSONL artifacts with the
ledger.

Focused tests:

```text
py -3 -m pytest tests/test_inaturalist_exposure_ledger.py -q
5 passed
py -3 -m ruff check scripts/audit_inaturalist_exposure_ledger.py tests/test_inaturalist_exposure_ledger.py
All checks passed
py -3 -m pytest -q
2,949 passed, 11 skipped
py -3 -m ruff check .
All checks passed
```

The inventory run was:

```text
py -3 scripts/audit_inaturalist_exposure_ledger.py --ledger C:\CodexArtifacts\D-CodexArtifacts\r2-inaturalist-source-global-ledger-20261111.json --cohort-root C:\CodexArtifacts\inat-birds-cc0-ascending-attempt-01 --inventory-root C:\CodexArtifacts --output C:\CodexArtifacts\pretrained-reuse-r2-local\inat-ledger-reconciliation-20261128-attempt-02\summary.json
```

The per-file follow-up used the same read-only command with a new output path,
`...\inat-ledger-reconciliation-20261128-attempt-03\summary.json`; its focused
tests remained 5/5 passing. Both outputs are preserved separately.

Both real-artifact audits produced exit code 2 as designed because history is
not reconciled. No iNaturalist source/media, legacy artifact, pre-existing
worktree, or running process was changed. No GPU or cloud compute was used.

## Next data gate

Do not expand this iNaturalist task or train on the 20-row remainder. A future
cohort needs a frozen source revision, sufficient genuinely new independent
groups after a complete exposure-ledger reconciliation, all required classes,
and a task where the paired modalities concern the same event. Resolve the
source-global history gap before interpreting additional live API counts.
