# R2 LSC50 timestamp-pairing metadata recheck (2026-11-14)

## Result

LSC50 remains `REVIEW_NOT_ADMITTED`; no media acquisition or model fit is
justified. The metadata-only recheck confirms that timestamp rows and video
triplets have matching aggregate sign/participant pair counts, but does not
establish a row-level video-to-IMU pairing convention. It also finds one
zero-duration timestamp interval.

## Pinned inputs and method

The audit parsed the already-preserved timestamp workbook and video ZIP central-
directory inventory. It did not read the ZIP payloads, any video, sensor signal,
or model artifact.

- Timestamp workbook SHA-256:
  `ee594ac0b35ee574fb6619363977468bf418c579aa0dbbf2d154f9b9fb068307`
- Video inventory JSON SHA-256:
  `7e192ef585779213772a4493ca8c4cedf96feaa1d68a23ec033a771323f7f815`
- Audit script SHA-256:
  `9299a2408905d409041a160e04995a0219c4ed42506521544e73fffa21b5f855`
- Machine-readable result:
  `C:\CodexArtifacts\pretrained-reuse-r2\lsc50-pairing-audit-20261114-attempt-01\result.json`
  (SHA-256 `39604b24478580b1f8dae4f6cf83dd296b7ccbda4b4b1901a6bc99fcb5b606fb`).

The workbook contains five participant-named sheets with 50 timestamp rows
each. Each sheet covers `N=0..49`, has unique N values and monotonic start
times. One row is invalid: sheet `Volunteer 5`, `N=49`, where end time equals
start time. The video inventory contains 4,000 AVI entries across four views,
1,000 unique `(sign, participant, repetition)` triplets, and 250
`(sign, participant)` pairs. Every video pair has four repetition IDs and all
1,000 triplets appear in each view. Video participant IDs are `0000..0004`;
workbook sheet numbers are `1..5`. Sign IDs span `0000..0049` in video names
and `N=0..49` in the workbook, but the inspected metadata and published loader
do not document the participant-number offset or which video repetition
corresponds to a timestamp row. Matching aggregate counts do not prove that
mapping.

The public timestamp worksheet values (`N`, sign name, start, end) were read
only to validate structure and timing; this was source metadata, not a model
evaluation. No official evaluation or sealed-audit split was opened. No model
inference or training occurred.

## Decision and reproducibility

The five-person limit already fails the provisional participant-group scale
for R2-M. The unresolved 1-based/0-based participant convention, four video
repetitions per sign/participant, single timestamp row per sign/participant,
and one invalid interval add a separate pairing/data-quality barrier. Do not
guess an alignment from row order or filename order. Leave the data unacquired
and the candidate unadmitted.

Reproduce with:

```powershell
python scripts/audit_lsc50_timestamp_pairing.py `
  --timestamps-xlsx C:\CodexArtifacts\pretrained-reuse-r2\lsc50-timestamp-structure-01\Timestamps.xlsx `
  --video-inventory-json C:\CodexArtifacts\pretrained-reuse-r2\lsc50-source-audit-02\VIDEOS.zip.inventory.json `
  --output-json C:\CodexArtifacts\pretrained-reuse-r2\lsc50-pairing-audit-20261114-attempt-01\result.json
```

Focused CPU tests: `2 passed`. Ruff: passed. The report and script only audit
source metadata; they do not admit the task or establish quality.
