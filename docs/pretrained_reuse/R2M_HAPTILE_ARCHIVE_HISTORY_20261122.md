# HapTile archive-history reconciliation follow-up — 2026-11-22

## Finding

A read-only comparison of the two pinned Hugging Face revisions found that
revision `917407abe013ac6a4c69544832900eeb2e879b28` had 39 task ZIPs. The later
source revision `54887384f1fe9e93b799e104194a582f9d8840b8` has 38: it removed
`Data/remove_suger_bag.zip` while retaining `Data/remove_sugar_bag.zip`. The
38 ZIPs present in both trees have identical Git object IDs. The canonical
archive also has the same LFS OID in both revisions.

Parsing only the final 65,536 bytes of each archive found 68 `trajectory.h5`
paths in the removed spelling-alias archive and 61 in the retained canonical
archive. Sixty-one internal names overlap; seven names occur only in the
removed archive. Names do not prove the HDF5 payloads are byte-identical or
that recordings are duplicates. The earlier tree therefore has 1,768
trajectory entries by central-directory count (1,700 in the pinned source
tree plus the 68-entry alias), while the current pinned source has 1,700.

This history does **not** reconcile the paper's 1,726 demonstrations with the
pinned release: 1,726 is 26 above the final source count and 42 below the
earlier tree count. The earlier alias's seven name-only entries are not a
validated explanation. No HDF5 payload, labels, model, validation, or sealed
audit were opened. The existing frozen split, conversion, corpus, features,
and checkpoints are unchanged.

## Reproduction

- Audit script: `scripts/pretrained_reuse/audit_haptile_archive_history.py`.
- Command:
  `python -m scripts.pretrained_reuse.audit_haptile_archive_history --output C:\CodexLocalArtifacts\pretrained-reuse-r2\haptile-source-history-20261122-attempt-03`
- External report: `C:\CodexLocalArtifacts\pretrained-reuse-r2\haptile-source-history-20261122-attempt-03\result.json`.
- Report SHA-256: `9f9ecb6a3a42fb5c9df1881e6c7ac4552b2d2e9bd5b5351a1cd35864a68c2edc`.
- Audit script SHA-256: `e2fdb93ed9538a1bb831974e2b67c7497aacc45a2b6273a6a1bbf22fdd0b764c`.
- API responses and exact range bytes are retained in that external attempt
  directory; both ZIP range requests returned 206 and fetched 131,072 bytes
  total. No archive payload was saved.
- Regressions: three parser tests pass. Ruff passes on the script and tests.

## Disposition

Keep HapTile `REVIEW_NOT_ADMITTED` for product-directed promotion. Its CC BY
release and task-held-out split support only the already documented
development pilot; operator identity remains unmapped, the paper/release count
remains unresolved, and the single laboratory scene cannot support
unseen-environment claims. Do not add examples from the removed alias to the
frozen corpus. The 26-record discrepancy is an explicit open source-provenance
question, not an inferred cleanup count.

Machine-readable receipt: [`haptile-archive-history-20261122.json`](../../manifests/pretrained_reuse/haptile-archive-history-20261122.json).
