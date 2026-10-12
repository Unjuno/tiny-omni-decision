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

## Input reader smoke

`quaternion_sto.parse_quaternion_sto` now preserves sensor column names, w/x/y/z
order, quaternion signs and actual processed timestamps (seconds converted to
milliseconds), without resampling or claiming video clock alignment. Nonfinite
values, malformed rows, nonincreasing times and nonunit quaternions are rejected.
Six CPU tests passed; the S40/A01/T01 training file parsed as (1817,5,4).

A second training header, S40/A05/T01, uses torso/humerus/radius sensors instead
of pelvis/femur/tibia. Sensor placement consequently reveals the upper/lower
activity group. A future 13-way classifier must include a placement-only prior
and within-placement comparisons; overall recognition cannot be interpreted as
pure motion understanding. These headers must not be silently mapped to the
same physical body locations. No quality training has been launched.

## Complete frozen training input audit

`scripts/audit_vidimu_training_sto.py` validated all 130 frozen training trials
against archive/split pins and saved every source member SHA256 and result to
`C:/CodexArtifacts/pretrained-reuse-r2/vidimu-training-sto-audit-01/result.json`.
All 130 passed strict parsing, finite/unit-quaternion and increasing-time checks.
All start at zero, have 20ms steps within floating-point tolerance, and span
11.30 to 69.96 seconds. Lower-body sensor layout occurs in 40 trials and upper
body in 90. No failed record was dropped; no validation/audit values were read.

This establishes usable processed training sensor inputs, not physical video
alignment or classifier quality. Keep different placements explicit, compare
against placement-only priors, and preserve all 13 classes. The initial zero-time
pose remains in the source; its role must be fixed in preprocessing before any
quality comparison. No arbitrary timestamp replacement or raw-row trimming is
authorized by this audit. The reduced-video download continues independently.
