# RoboMNIST residual group exposure recheck — 2026-11-10

## Decision

Do not use the previously uninspected RoboMNIST residual groups for a new validation run. The metadata-only audit found 323 of 345 groups in two earlier split manifests. The remaining 22 groups can form at most six same-condition pair units (24 balanced rows), far below the project's validation floor. They are not sufficient to support another model-quality evaluation.

No media, feature cache, model weights, sealed-audit file, or GPU was accessed. No model was trained or evaluated.

## Method and pinned inputs

The audit script reads only the 60 `directory-*.json` source inventories under `robomnist-all-directories-audit-20261103-attempt-01`, checks each file and `config.json` against the frozen split manifest, verifies the prior non-sealed candidate pool's train/validation hashes and metadata-only flags, and scans bounded project-owned RoboMNIST receipts. Paths containing `sealed` and earlier outputs from this residual audit are skipped. It does not open source media, model artifacts, feature tensors, or sealed audit inputs.

- Source worktree commit: `e07e049ff4387484fb754e0fff963631f3cc7c85`.
- Frozen source manifest SHA-256: `3f4575ba41d944310e1de83cbdc68385390581723262245b97dc0d2c71db660f`.
- Audit script SHA-256: `3b735b5f1d57d61078208a3717c32b6d8baa7a581ad90bc97fb0e7c9d1b04993`.
- Source inventory: 2,135 group IDs; canonical sorted-ID SHA-256 `0446701eb3634ead8e34ac2284918c2b339c31ece4e846da9d6e27dd8aef0b88`.
- Prior non-sealed candidate pool: 1,790 IDs; canonical sorted-ID SHA-256 `26107609df3851e8e976dd9f1dc54b5ef4a53b61608148af94a23a351a0a5b13`. Train SHA-256 `7d37ef856d0e6b8c3d548d74e8e924224b76c46fffcd52a2db3129c095e31f86`; validation SHA-256 `b4029fe384665408bab9d144f6ffd12b82715714649b98d0a2688ccf0722e970`.
- Existing residual-screen manifest: `3f4575ba41d944310e1de83cbdc68385390581723262245b97dc0d2c71db660f`.
- Earlier split manifests: attempt 01 SHA-256 `dccb08b05b880f0d66180dbc54e6602d0ec329a0edb51c7b5de217deaa65bdce`; attempt 02 SHA-256 `9118fe36ed41a038900899fde9625573e805729b42ff82271f2912ff1037ddcf`.
- Result JSON: `C:\\CodexArtifacts\\pretrained-reuse-r2\\robomnist-residual-exposure-audit-20261110-attempt-03\\result.json`, SHA-256 `1a859e2d509fabd1c300712f3285968dfb74a387e8adaf576c2dbb3ac5ee5931` (60 records scanned).

Two earlier metadata-only audit outputs were retained rather than overwritten. Attempt 01 used an older inventory whose config hash did not match the frozen manifest and is not evidence. Attempt 02 used the pinned inventory but counted the earlier audit's own output as an input receipt; attempt 03 excludes prior outputs and is the reported result.

The earlier attempt 01/02 split manifests do not have adjacent run summaries in their artifact directories. Their presence proves metadata allocation only; whether those attempts opened media, trained, or ran inference is UNKNOWN. The audit therefore does not label all 323 groups as model-exposed, and it also does not treat them as untouched.

## Results

| Measure | Result |
|---|---:|
| Total source repetition groups | 2,135 |
| Prior non-sealed group pool | 1,790 |
| Residual groups outside that pool | 345 |
| Residual IDs present in prior non-sealed split manifests | 323 |
| Residual IDs absent from the scanned non-sealed records | 22 |
| Same-condition pair units possible from those 22 | 6 |
| Balanced examples at four rows per pair unit | 24 |

The 22 not found are only “not observed in the scanned records”; absence is not proof of project-wide non-exposure. At 22 groups, this subset fails even a 200-independent-group floor. It also fails the RoboMNIST-specific 1,000-group validation requirement from the episode-task feasibility gate.

## Interpretation and next action

This closes the residual-pool question without a new model run. Do not rerun or retune the AV-match readout on these records, and do not claim a fresh or blind evaluation. Keep the existing AV-match result as the narrow synthetic digit AV-correspondence diagnostic it is.

Continue R2 by admitting a different rights-cleared, independently grouped Decision task through the existing gates. If no source passes, keep the R2 gate open and continue targeted source discovery; do not synthesize extra pairs to reach the sample floor.

## Verification

- `python -m ruff check scripts/audit_robomnist_residual_exposure.py tests/test_audit_robomnist_residual_exposure.py`: pass.
- `python -m pytest tests/test_audit_robomnist_residual_exposure.py -q`: 2 passed.
- The tests cover hash mismatch fail-closed behavior and verify that sealed-path records are ignored.
