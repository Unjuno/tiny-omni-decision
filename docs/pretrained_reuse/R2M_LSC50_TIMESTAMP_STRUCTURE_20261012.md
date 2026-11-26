# LSC50 synchronization structure: repetition mapping unresolved

CRC-checked extraction of `IMU/INFO/Timestamps.xlsx` succeeded using bounded
range reads. SHA256 `ee594ac0b35ee574fb6619363977468bf418c579aa0dbbf2d154f9b9fb068307`.
The workbook contains5 volunteer sheets, each A1:D51 with headers N, Sign,
Time Start(s), Time End(s). Header/dimension inspection only; no timing rows
were reported or used to pair examples. Sensor/video contents remain unopened.

The [primary descriptor](https://www.nature.com/articles/s41597-024-04172-5)
defines video filename fields as sign, volunteer and repetition. It describes
synchronization using intervening pauses and a fixed starting/ending posture,
with IMU segmentation timestamps. This is not evidence of hardware-level
frame synchronization. Its250 sign/volunteer IMU measurements contrast with
1000 trials per camera view. The actual workbook has no repetition column.

Do not duplicate one IMU interval across4 video repetitions and call them paired
simultaneous observations. The mapping of zero-based video volunteer names to
one-based IMU sheet names also needs explicit verification, not arithmetic guess.
This source passes rights readiness, but sample-level pairing remains UNKNOWN.
No participant split or training is claimed yet. Next inspect upstream loader
documentation for repeat/clock conventions; if not documented, reject this source
for synchronized-fusion claims while retaining it as an unpaired research option.

Metadata: `C:/CodexArtifacts/pretrained-reuse-r2/lsc50-timestamp-structure-01`.
[Exact headers and hash](R2M_LSC50_TIMESTAMP_STRUCTURE_20261012.json).
The local environment lacked openpyxl; existing standard-library ZIP/XML parsing
was used without upgrading dependencies. No heavy data download or GPU work.
