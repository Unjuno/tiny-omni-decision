# VIDIMU training-only format inspection

After the subject split was frozen, inspected only its lexicographically first
training raw member: S40/A01/T01. No validation/audit payloads were read.
Script: `scripts/inspect_vidimu_training_record.py`.
Result: `C:/CodexArtifacts/pretrained-reuse-r2/vidimu-training-format-audit-01/result.json`.
Member SHA256: `acd69a5fabf570fc18d6fe0163241ff14cdb4d805aceeb50392bfe29f1a46990`.

Fields are QUAT,w,x,y,z,timestamp. Five device labels (qsHIPS, qsRUL, qsRLL,
qsLUL, qsLLL) each have 14,142 rows, including one zero timestamp. Numeric values
are finite; quaternion norms are approximately one. Positive timestamps are
strictly increasing within each device, spanning 36.32456 seconds. Median raw
timestamp interval is approximately 0.00299 seconds, not 0.02 seconds. This is
not sufficient to infer the physical acquisition rate or time-base accuracy.

The upstream synchronization code assumes 50Hz IMU when converting video frame
offsets into row counts. Raw timestamp semantics and processed .mot/.sto time
bases therefore need reconciliation before using those row cuts. Do not replace
the raw timestamps with an assumed 50Hz grid, or claim synchronization from
matching trial names. The initial zero timestamp's semantic role remains
unverified. Do not feed these quaternion channels into the earlier acceleration
sensor model without an explicitly different input contract.

This observation changes the next step to raw/processed time-base verification;
it does not establish a dataset defect, reject all VIDIMU trials, or produce a
model-quality result. The analysis archive download is still running.

## Processed time-base follow-up

`scripts/inspect_vidimu_timebase.py` inspected only the same training trial's
original .sto and .mot. Both contain 1,817 rows, 0 through 36.32 seconds with
0.02-second spacing. The .sto explicitly declares DataRate=50. Thus the 50Hz
assumption is supported for these processed records, but not raw row indices.
Their duration agrees closely with raw positive timestamps (36.32456 seconds).
This narrows the issue to raw/processed sampling conversion and initial-row
handling, rather than demonstrating a global recording-duration discrepancy.
Results and member hashes:
`C:/CodexArtifacts/pretrained-reuse-r2/vidimu-training-timebase-audit-01/result.json`.

The first analysis download exited with a requests read timeout; its output is
preserved. A separate IPv4 curl download is active in
`C:/CodexArtifacts/pretrained-reuse-r2/vidimu-analysis-audit-02`, using HTTPS
verification, 15-second connection timeout, 600-second total limit and a
60-second low-speed limit. No data from that archive is trusted until the same
pinned size/MD5 checks pass. The failed process was not killed or restarted.
