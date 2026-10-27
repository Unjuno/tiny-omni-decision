# Source screen: ambient audio, vibration and environmental measurements

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
