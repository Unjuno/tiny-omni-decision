# NEMAR on006334 AVspeech source screen (2026-10-26)

## Decision

Do not admit NEMAR `on006334` as the natural audio/video input corpus for the
Path A decision comparison. The study task is genuinely audiovisual, but the
published dataset package contains behavioral logs and physiological/anatomical
data rather than the audiovisual media that the model would need to consume.
This is a metadata-only screen: no MEG, MRI, behavioral, or stimulus files were
downloaded; no model was run.

## Evidence

- NEMAR identifies version `v1.0.0`, license `CC0`, 30 participants, a size of
  171.9 GB, and modalities `ANAT` and `MEG`. Its README describes audiovisual
  synchrony judgments during encoding and auditory-speech retrieval cued by a
  speaker's face. That makes the human task AV, but does not establish that the
  recorded AV stimuli are in the package.
- The pinned version manifest has 1,201 entries. A path/extension scan found no
  `stimuli/` directory and no standalone video or audio assets (`mp4`, `m4v`,
  `mov`, `avi`, `mkv`, `webm`, `wav`, `mp3`, `flac`, `m4a`, or `aac`). It lists
  128 MEG event TSVs (2,611,848 bytes total), 30 `Movies.mat` files (100,986
  bytes total), and 30 `Sounds.mat` files (100,771 bytes total). The metadata
  therefore supports event/behavior analysis but not a direct frozen audio/video
  decision model input pipeline.
- The public `participants.tsv` contains participant identifiers, age, and sex.
  It was inspected from the official endpoint but not retained with the source
  snapshot or copied into the repository. The full human-subject data is far
  larger than the task needs and was not acquired.

## Reproducibility record

All source references below are pinned to NEMAR version `v1.0.0`. Small source
metadata files are retained under
`C:\CodexArtifacts\pretrained-reuse-r2\nemar-on006334-source-screen-attempt-01`;
their SHA-256 values are:

| File | SHA-256 | Bytes |
|---|---|---:|
| `manifest.json` | `dc8226a796ae59ebfbb8173da6d702d3e6c0d0962865e996651c1f2aef70e8bb` | 498,835 |
| `dataset_description.json` | `71bc02e39cbf67f38c8de83b96b4711ab516c5bbfd92a1fb3d40f68a9471019a` | 1,267 |
| `README.md` | `1778a301144aa690deda36e29c28d112676ad85e2ea1378773f088b74af3d001` | 4,119 |

## Disposition

Keep the source screen as a rejected input-corpus candidate. CC0 licensing does
not compensate for missing AV input assets or make a MEG task equivalent to the
target model's media decision task. Reconsider only if the original stimulus
assets are published under compatible terms and can be mapped to independent
participant/trial groups without acquiring unrelated sensitive data. Until
then, R2's rights-cleared, both-input-required natural AV gate remains open.

## Official sources

- [NEMAR on006334 dataset page](https://ww2.nemar.org/dataset/on006334)
- [Pinned v1.0.0 dataset description](https://data.nemar.org/on006334/v1.0.0/dataset_description.json)
- [Pinned v1.0.0 file manifest](https://data.nemar.org/on006334/v1.0.0/manifest.json)
- [Pinned v1.0.0 README](https://data.nemar.org/on006334/v1.0.0/README.md)
