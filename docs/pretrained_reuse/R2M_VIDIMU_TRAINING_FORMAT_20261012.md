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
