# Source screen: ambient audio, vibration and environmental measurements

## Completed bounded archive preflight

After the source-only entry below, the predeclared smallest archive
`23_take-from-stove.zip` was fetched and its 69,152,997 bytes matched the
publisher MD5. It contains five HDF5 sensor files. The SHA-256-pinned archive
was opened locally; all five files contain **368 participant/trial pairs from
9 participants**, with the same pair identities across the five sensors.
These are 368 events represented five times, not 1,840 independent events.
Each sensor's participant/trial pairs are unique. This is one activity from
one archive and does not establish support or splits for the other 24 classes.

For matching participant/trial pairs, maximum absolute timestamp differences
from sensor 1 were 0.2210, 0.2030, 0.2500 and 0.2300 in the stored native
units for sensors 2–5, respectively. This is an exact comparison of saved
numeric values, not proof of common physical clocks or units. Vibration rate
values span 1,075.42–1,094.53 across the five files; use recorded per-row rates,
not an assumed single rate. One exploratory first row was inspected for array
shapes: audio had 42,605 int16 values, each vibration axis 2,876 int16 values.
Those shape observations do not prove waveform alignment or a sampling rate.

The preflight took 474.40 seconds including network, checksum and initial
single-file schema inspection. CPU/h5py 3.14.0 was used; no encoder, GPU,
training, validation or sealed audit ran. The saved archive remains outside
Git under
`C:/CodexArtifacts/pretrained-reuse-r2/adl-audio-vibration-archive-20261027-attempt-01/`.
Its SHA-256 is `04bf32f55c93f54c49525d2b0581cb51f14c1ec302afb56b70a995fa4aa8f706`;
the publisher MD5 is `10834b9e0813ee9441abe2c3f5c2233a`.

The [identity receipt](R2_ADL_SENSOR_IDENTITY_20261027.json) records each
member hash, participant counts, pair matches and rate ranges; its SHA-256 is
`9adfbeb513acf5c9855fa94be0fbeec39ad047d7f7ac4bd9bfb680ab5b3ffb6a`.
The original download/schema result SHA-256 is
`f2dc538461d5726a0019cbad6802b561c4c94a48f20b0f3b42a3e70983bd9f50`.
The original bounded acquisition script is preserved outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2/adl-audio-vibration-archive-preflight-20261027.py`,
SHA-256 `47b8b63adf42f6964c5f3f26ca257bb93c5dcaa5731c661f3fe9cd2451cfc2ae`.
The identity checker can be reproduced, using the existing h5py environment,
with a **new** output filename:

```powershell
& C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv/Scripts/python.exe `
  -m scripts.audit_adl_sensor_identity `
  --archive C:/CodexArtifacts/pretrained-reuse-r2/adl-audio-vibration-archive-20261027-attempt-01/23_take-from-stove.zip `
  --output C:/CodexArtifacts/pretrained-reuse-r2/adl-audio-vibration-archive-20261027-attempt-01/sensor-identity-NEW.json
```

Tests exercise identity matching across reordered rows, duplicate/nonfinite
rejection and unmatched participants/trials. Ruff passed; the local CPU suite
reported **2,633 passed, 11 skipped**. More activities and a fixed
participant-disjoint task are required before any model comparison. The
14-person, single-environment limitation remains; this does not close the
natural joint-AV or final product gates.

## Disposition

This is a possible **audio + sensor research task**, not an audio/video dataset
and not an admitted product-quality benchmark. The producer describes 8,615
activity instances performed by 14 people in one kitchen/dining environment.
The five device recordings of each instance are dependent observations; they
are not 43,000 independent activities. Participant-disjoint evaluation has at
most 14 participant groups, before splitting, and cannot meet the roadmap's
provisional 200-independent-group quality screen.

Use this as a scoped feasibility lead only. Prior to any fit, verify the actual
file and label schema, the participant/session identity needed for splitting,
input synchronization, cross-split content overlap, and the task contract.
Matched audio-only, vibration-only and joint models, plus missing/shuffled input
controls, are required to establish whether both inputs add useful information.
Environmental or device identifiers must not silently become label shortcuts.
Do not present source-reported sample counts as independent test groups.

## Primary source evidence

The [versioned Zenodo release](https://zenodo.org/records/7937591), DOI
`10.5281/zenodo.7937591`, publishes activity-labelled recordings of ambient
audio, vibration, infrared arrays and low-frequency environmental measurements.
It provides 25 activity categories as separate archives and lists a checksum
and size for each. No video is listed as an input; the recording paper describes
live cameras used by an observer, not video in the released data.

The [authors' dataset paper](https://www.mdpi.com/2306-5729/9/12/144)
states that each activity instance was recorded by five multisensors, describes
14 volunteers, two supervised sessions per participant for a complete set, and
states CC BY 4.0 as the dataset license. The [original paper PDF](https://pdfs.semanticscholar.org/ef1d/0bd7a9eba961203a601f7723274219868409.pdf)
reports 8,615 activity instances. The archive page reports over 43,000 device
samples, consistent in scale with five recordings per activity; neither count
is an independent-person count. Manufacturer clocks, raw timestamp behavior,
label correctness and full file-level integrity remain unverified locally.

The study reports that recordings were pseudonymized and participants were
informed of intended publication. Those statements are source evidence, not
an independent privacy or downstream-use assessment.

## Actual access and limitations

The public Zenodo record and indexed publisher descriptions were read on
2026-10-27 JST. The local Zenodo API confirmed version `1.0`, 29 files, and
dataset license `cc-by-4.0`. Its 5,452-byte README was downloaded and matched
the publisher MD5 `61f46b9339b227c4571a796dff9bd73b`.

The README says the data are stored by activity and sensor in HDF5 files. It
lists `participant`, `trial`, `timestamp` and `vibration_sample_rate` columns,
with audio and three vibration axes stored as ragged arrays. This is a feasible
split/loader lead, not proof that the actual columns, IDs and times are correct
or sufficiently independent. It also says infrared/light measurements have
separate timestamps; do not assume those streams are already row-aligned.
The browser-backed API request failed, while the local API and README retrieval
completed. A separate data license was checked; the paper explicitly identifies
the dataset as CC BY 4.0.

Original downloaded metadata and README are retained outside Git in
`C:/CodexArtifacts/pretrained-reuse-r2/adl-audio-vibration-source-20261027-attempt-01/`:

- Record SHA-256: `c01bf60460981280c5063ea54c038d41c751bf3056220059c3896cbc5d2b97c6`.
- README SHA-256: `d5fe3c9097226f00656a65b86d8374a4bf62c42f8c80b156cbe2a69ab33f83ac`.

A separate predeclared preflight is acquiring the smallest source archive,
`23_take-from-stove.zip` (69,152,997 bytes), to inspect HDF5 structure. It was
still in progress at the time of this entry. Its output directory and complete
protocol are `C:/CodexArtifacts/pretrained-reuse-r2/adl-audio-vibration-archive-20261027-attempt-01/`.
An archived publisher MD5 is required before inspection; a 1 GiB free-space
reserve and 128 MiB uncompressed member cap apply. No source archive had been
successfully acquired at this entry; no model weights or sealed data were
opened and no training occurred.

The complete archive set is listed as approximately 4.6 GB; local C: free space
was approximately 1.35 GB during the preflight. Do not download the whole set
without checking current capacity. A future bounded source check may select a
small archive in advance, verify its published checksum, and inspect metadata
without selecting or tuning a final evaluation set. This source cannot alone
resolve the required text/image/audio/video and natural joint-AV quality gates.

The user explicitly declined VisTouch acquisition; do not fetch its media.
UnAV-100 remains excluded pending rights resolution. No product promotion or
quality claim follows this source screen.
