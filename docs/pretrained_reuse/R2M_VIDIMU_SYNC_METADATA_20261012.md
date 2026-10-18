# VIDIMU synchronization metadata: verified acquisition, two ambiguous trials

IPv4 curl acquisition completed successfully in 282 seconds. Archive size
169,608,328 bytes and publisher MD5 matched before opening the ZIP.
Archive SHA256: `7200b12bdce559c08ccf430761791d5a794f11dc304b0a5152d8e41f45e7eca9`.
Extracted `analysis/videoandimusync/infoToSync.csv` SHA256:
`4596803e90e2908717ab227846f7a1c10a5c6b29c9bcbddb90e940a7259c519f`.
Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/vidimu-analysis-audit-02`.
The failed first download is preserved. No observation payloads were extracted.

The CSV contains 366 rows for 181 subject/activity/trial identities: 149 mp4,
149 csv, 34 raw, 34 mot entries. Two identities have positive corrections for
both sensor and video, matching the previously identified code-path ambiguity:

| Trial | Raw/mot CutFrames | Video/csv CutFrames | Frozen group |
|---|---:|---:|---|
| S57 A07 T01 | 14 | 11 | validation |
| S53 A13 T03 | 1 | 2 | train |

These offsets are expressed in the synchronization code's 30Hz grid. They are
not automatically additive or safe to apply to raw row indices. Published
TheoRmse does not identify which direction won because both entries receive
the selected comparison's RMSE. No correction direction is inferred here.

Next reconcile CSV identities against the frozen original inventory, including
trials with no CSV entry, and compare original/processed/corrected timing on
training data. Do not silently drop ambiguous validation trials or refit their
offsets to improve quality. The task remains an alignment-readiness audit, with
no fusion training or evaluation completed.

Verification command:
`python -m scripts.audit_vidimu_analysis --verify-existing C:/CodexArtifacts/pretrained-reuse-r2/vidimu-analysis-audit-02`.
Current regression suite: 500 passed, 1 skipped; Ruff passes. Commit befee5a CI
passed both checks. These checks do not establish physical synchronization.

## Full identity reconciliation

`scripts/reconcile_vidimu_sync.py` verifies the pinned split/CSV hashes, unique
offset types, 1-14 frame bounds, paired raw/mot and mp4/csv agreement, and that
all CSV trial identities exist in the frozen inventory. It succeeded.

| Group | Single direction recorded | Missing record | Both directions |
|---|---:|---:|---:|
| train | 114 | 15 | 1 |
| validation | 35 | 3 | 1 |
| sealed audit (metadata only) | 30 | 9 | 0 |

Full per-trial statuses are preserved at
`C:/CodexArtifacts/pretrained-reuse-r2/vidimu-sync-reconciliation-01/result.json`.
No missing entry was interpreted as zero or any trial removed. Single-direction
metadata alone is also not proof of physical synchronization.

The pinned reduced-video archive (1,060,471,682 bytes; publisher MD5
`b90c4ae792b3dd54a8ba337ae5f75e36`) is being downloaded separately to
`C:/CodexArtifacts/pretrained-reuse-r2/vidimu-small-video-audit-01` over IPv4 HTTPS.
Its contents are not yet verified or decoded. Full checksum verification and
trial pairing must precede any training/validation feature extraction. Audit
payloads remain unopened. Whole-trial recognition, if evaluated, must be labeled
as episode-level fusion and cannot establish frame-synchronized causal decisions.
