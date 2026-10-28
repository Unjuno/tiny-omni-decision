# R2 ADL full sensor-identity and split-support audit

## Result

The full pinned source audit passed archive authentication and inspected all
five sensor HDF5 members for all **25 activities** (**125 members**). The five
sensors in each activity had the same participant/trial identities and row
counts. No duplicate participant/trial identity was found within a sensor.
Across all activities, the fixed participant split has **12 train and 2
validation participants**, with **zero participant overlap**.

The source has **8,615 activity rows**: 6,412 assigned to train, 1,972 to
validation, and 231 unassigned. **24/25 activities** have nonempty train and
validation splits. All 231 rows in `00_no-activity` use the source's special
participant ID 999 and remain unassigned under the preregistered protocol;
that activity therefore fails coverage. No rows were dropped or duplicated.

Class support is also highly uneven: `14_sweep` has 9 rows (6 train / 3
validation) from 7 participants; `11_typing` has 14 (10 / 4) from 9;
`15_vacuum-cleaning` has 14 (11 / 3) from 8. These validation counts are far
too small for credible per-class quality claims.

For matched rows, the largest sensor-to-sensor timestamp difference was
**0.376 native units** and the largest sample-rate difference was
**25.4513**. These comparisons are diagnostic values only: the dataset's
physical timestamp units, clock relationships and cross-activity session
independence are not established. Preserve the measured per-row/per-sensor
rates; a single shared sensor rate is not supported by the source.

## Pinned artifacts and replay

- Source record SHA-256: `c01bf60460981280c5063ea54c038d41c751bf3056220059c3896cbc5d2b97c6`
- Download plan SHA-256: `a387e837d9e14ea00fec20e72c07331fafd3f12e07661cf671da0f4216152dc2`
- Download receipt SHA-256: `af284fe02e447a33194aa0fa871a0601dc0f067ec81137ceb9a17843379230aa`
- Fixed participant split protocol SHA-256: `9c301d4a3acc37711d25943e7165533410bb3297630eef19c7f6431d2daaf5f8`
- Audit result SHA-256: `1cff63f035b97737224c45515386e84f01a20fd2fd81f676d817a44d39546a3f`
- Audit script SHA-256: `b48373d9b3e99b4b790eddd1af030071958ec7b3852095efedcc79545f08d414`
- Focused tests SHA-256: `217039cc48371f4e908b9ee0c555b07dcc443253abe558c2fcd8539ec50f9a2f`

The complete aggregate result, including each archive and member digest and
per-activity counts, is [R2_ADL_FULL_SENSOR_IDENTITY_20261028.json](R2_ADL_FULL_SENSOR_IDENTITY_20261028.json).
It contains no participant IDs, only aggregate group counts. The external
replay output was written separately under
`C:/CodexArtifacts/pretrained-reuse-r2/adl-full-sensor-identity-audit-20261028-attempt-02/`.
Attempts 01 and 02 produced byte-identical JSON. The replay validates the
download-plan and receipt hashes, rechecks every archive size/publisher MD5 and
SHA-256 against the acquisition receipt, validates the split protocol, then
validates every ZIP member's size and CRC while computing a local SHA-256.

HDF5 members were decompressed one at a time into temporary external files and
removed after each read. Only participant, trial, timestamp and vibration
sample-rate datasets were read; audio and vibration waveform arrays were not
read. Python 3.11.16, h5py 3.14.0, NumPy 1.26.4 on Windows 10 were used. CPU
only; no GPU, model fitting, inference, validation scoring or sealed-audit
access occurred.

## Decision and limits

This audit improves source integrity evidence but **does not admit ADL for a
quality experiment**. One activity has no assigned split, several activities
have only 3–4 validation rows, the source has only two validation participants
in one environment, and it contains no released video. Shared row identity and
stored timestamps do not prove physical synchronization, recording-session
separation, or absence of temporal leakage. The objective still requires a
task where both inputs are necessary and a defensible, much larger independent
evaluation set. Do not train on this corpus or infer broad audio/sensor quality
from these metadata results alone.

