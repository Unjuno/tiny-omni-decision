# Physion++ train/development split integrity — 2026-11-28

## Scope

This read-only audit checked the existing `physionpp/readout` train and
development-validation manifests for split leakage. It inspected record
metadata and hashes only: no video bytes were decoded, no model was loaded, no
model metric was computed, and the publisher's final test/sealed data was not
opened. The validation labels are now exposed development data and must not be
described as a blind audit.

The source revision is `0c53b983ca138962ba636c0936e735490ad4c326`. The corpus
manifest declares seed 17, 615 source-asset groups, and 123 groups in the
development split. The source-asset identity is the recorded scenario plus
scene seed. This is an in-domain random group split: all ten scenarios occur
on both sides, so it does not measure scenario-held-out generalization.

## Results

| Identity check | Train | Development validation | Shared |
|---|---:|---:|---:|
| Records / unique record IDs | 652 | 148 | 0 |
| Unique source record IDs | 652 | 148 | 0 |
| Source-asset groups (scenario + seed) | 492 | 123 | 0 |
| Media paths | 652 | 148 | 0 |
| Media SHA-256 identities | 652 | 148 | 0 |
| Normalized Decision content fingerprints | 652 | 148 | 0 |

The repository's `check_train_eval_splits` implementation also returned
`disjoint` with zero shared source IDs and zero normalized-content fingerprints.
Its fingerprint normalizes the state, question and options and includes media
identity. Repeated task wording alone is expected in this fixed binary task;
the check confirms that it is not paired with a repeated video asset across the
two splits.

The corpus manifest independently records the same train/validation counts,
zero shared source IDs/assets/media/content fingerprints, seed 17, and a
validation-group-order SHA-256 of
`f46ee3fdce6fbc24b56220fd6d5d5ca34e82b9bb019c0096f959ec746611f546`.
The local check reproduced those disjointness properties from both JSONL files
using the repository's typed schema and content-fingerprint function.

## Interpretation and limits

This closes the specific source-asset leakage concern between the existing
train and development-validation partitions. It supports using this validation
split for development checkpoint selection in a separately frozen Path A
experiment. It does not provide a blind or independent result, does not verify
the publisher's final-test grouping, and does not establish video-label
observability or task quality. The earlier train-only factor and temporal
sampling diagnostics remain relevant. Do not tune or report a final-test claim
from this development split.

## Reproduction

```powershell
python scripts/audit_physionpp_split_integrity.py `
  --train C:\CodexArtifacts\tqpp\fullclip-v1\train.jsonl `
  --validation C:\CodexArtifacts\tqpp\fullclip-v1\validation.jsonl `
  --output C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-split-integrity-20261128-attempt-03\audit-report.json
```

- Script: `scripts/audit_physionpp_split_integrity.py`
- Script SHA-256 (LF-normalized source):
  `8ee26c388c41113108f643e91baff793159ccfd2a9204357942bfad752188331`
- Tests: `tests/test_audit_physionpp_split_integrity.py` — 8 passed
- Ruff: passed for the audit script and its tests
- Train JSONL SHA-256:
  `0de1bbbb1cc54000936c1771ea54e375c214fedda36734536f31a44c4f64932b`
- Development-validation JSONL SHA-256:
  `1254f01256197f1c2e23f8fa13dc03f87789c328f32bf768300134dd6b70d8dc`
- Corpus-manifest SHA-256:
  `49abdfe672c047ac5578309eab35e0a6e05ca5da13a0ddd939f7ab503819892b`
- Machine-report SHA-256:
  `64eee5a53150bd3ff4e1db0b214e7a5af709ef4444f5655835b4100c8bc4d95e`
- Output: `C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-split-integrity-20261128-attempt-03\audit-report.json`
