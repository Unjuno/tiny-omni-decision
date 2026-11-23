# Urbansas / MAVD source-task screen — 2026-11-23

## Disposition

**Source rights: the MAVD subset is identified as CC BY 4.0. Task fit: promising
for a narrowly defined audio-visual source-localization diagnostic. R2 quality
admission: NOT ADMITTED.** The usable MAVD subset has only five acquisition-
location IDs in one city, no official independent Urbansas partition, and a
large fraction of audio events that are ambiguous when matched to visible
tracks. It cannot support a robust independent quality estimate or product
quality claim under the current gate.

The task has a plausible genuine cross-modal structure: the published method
uses timed audio annotations to decide which matching-class video tracks are
audible, then uses video boxes to approximate source location. Audio alone
does not provide a visible target position, while video alone does not identify
which vehicle is producing the sound. This is a task-design argument from the
publisher's paper, not a measurement from this project.

## Rights and scope

- The [Urbansas Zenodo v1.0 record](https://zenodo.org/records/6658386)
  states that its annotations are CC BY 4.0 and that source-media terms follow
  the upstream releases.
- The included [LICENSES.txt](https://zenodo.org/records/6658386/files/LICENSES.txt?download=1)
  identifies the MAVD subset as CC BY 4.0 and the TAU subset as non-commercial.
  Only rows whose video-annotation `subset` is exactly `mavd` were counted.
  No TAU or unpublished-material media were acquired.
- The original [MAVD-traffic Zenodo record](https://zenodo.org/records/4741232)
  describes synchronized traffic audio/video and four Montevideo recording
  locations. Its published archive provides train, validation, and test
  files, but the Urbansas annotations used here do not declare a split that
  would establish independent location-held-out groups.
- The whole Urbansas release is about 9.7 GB and bundles mixed-source media.
  No audio/video/still media were downloaded. No model was loaded; no feature
  extraction, training, or model evaluation was performed.

## Annotation-only inspection

Only the 731,143-byte official `annotations.zip` was downloaded. Its Zenodo
MD5 matched `02a1ce5a50857aa46dfceb2f43279c76`; local SHA-256 is
`5d1266f040f0ed9379e2396ad185128007b4a178e000d54df1ef8fd427034c50`.
Extracted annotations are outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2-local\urbansas-source-screen-20261123-attempt-01\`.
The machine-readable count receipt is `source-screen-summary.json`, SHA-256
`d8f9193a4b7fd6a3cab59f3f504b999128e809c96c12d499e976a1401e03ba00`.

| MAVD-only label evidence | Count |
|---|---:|
| Video annotation rows | 15,639 |
| Distinct annotated clips | 540 |
| Audio annotation rows / clips | 859 / 540 |
| Acquisition-location IDs | 5 |
| Audio vehicle events (excluding clip-level and `offscreen` rows) | 663 |
| Events overlapping exactly one same-class video track | 313 |
| Events overlapping multiple same-class video tracks | 296 |
| Events with no same-class video track at an annotated frame | 54 |
| Separate `offscreen` audio events | 151 |

“Overlap” here means the audio event's `[start, end]` interval contains at
least one annotated video frame time for a same-class track. The exact-track
counts are a deterministic annotation join, not a new performance metric.
For 296 events the available labels do not identify one unique visible track;
for 54 events no matching track is present at a sampled annotated frame.
These rows would require a predeclared ambiguity/abstention rule before any
task construction. Do not discard them after model predictions.

The five IDs and clip counts are:

| Location ID | MAVD clips |
|---|---:|
| `acevedo0103` | 100 |
| `magallanes0308` | 175 |
| `rivera0922` | 30 |
| `rivera0923` | 145 |
| `templete0103` | 90 |

All five are in Montevideo. `rivera0922` and `rivera0923` may be the same
physical recording area on different dates; treating them as separate groups
would be optimistic. The upstream record describes four physical locations,
so the effective independent site count may be four rather than five.

## Decision and next use

Keep as a **diagnostic candidate only**. Do not train on it as evidence of
general audio-video Decision quality. The small number of independent sites
prevents a credible site-held-out train/validation/sealed design; the task
also needs a frozen mapping from ambiguous audio/video overlaps to decision
options. The candidate can be reconsidered only if a source-compatible
location-held-out protocol can be fixed without touching a final test and the
ambiguity rule is defined before fitting. A small pilot, if ever authorized,
would prove task wiring only and must not count as R2 quality admission.

This source screen does not change the roadmap's requirement for a stronger
rights-cleared synchronized AV task with enough independent groups, nor does
it close any modality or fusion gate.

## Primary references

- [Urbansas Zenodo record, v1.0](https://zenodo.org/records/6658386)
- [Urbansas included license file](https://zenodo.org/records/6658386/files/LICENSES.txt?download=1)
- [MAVD-traffic Zenodo record](https://zenodo.org/records/4741232)
- [Fuentes et al., Urban Sound & Sight, ICASSP 2022](https://sightsound.org/papers/2022/Fuentes_Urban_Sound_%26_Sight_Dataset_and_benchmark_for_Audio-Visual_Urban_Scene_Understanding.pdf)
