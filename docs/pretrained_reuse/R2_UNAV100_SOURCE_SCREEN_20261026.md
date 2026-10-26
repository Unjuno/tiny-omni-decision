# UnAV-100 audio-visual source screen — 2026-10-26

## Scope

Read-only license/task/split preflight for UnAV-100 as a possible natural
audio-video Path A task. Only the official split CSV was downloaded; no event
annotations, extracted features, source videos, or media references were
opened. No model was run or trained.

Pinned sources inspected:

- Project site: [UnAV-100](https://unav100.github.io/).
- Official code: [`ttgeng233/UnAV@1609deba4fbaf25222c6ee6c7d43b0d0ddfab20d`](https://github.com/ttgeng233/UnAV/tree/1609deba4fbaf25222c6ee6c7d43b0d0ddfab20d).
- Hugging Face dataset card: [`ttgeng233/UnAV-100@daeb88bc14eeff48f8a4ac2836b67f6c350f847b`](https://huggingface.co/datasets/ttgeng233/UnAV-100/tree/daeb88bc14eeff48f8a4ac2836b67f6c350f847b).
- Published dataset paper: [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Geng_Dense-Localizing_Audio-Visual_Events_in_Untrimmed_Videos_A_Large-Scale_Benchmark_and_CVPR_2023_paper.html).

## Dataset and split metadata

The project describes dense localization of audio-visual events across 100
categories in more than 10,000 untrimmed videos. It publishes a CSV whose rows
contain YouTube ID, start time, end time and train/validation/test assignment.
The official CSV was saved outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2/unav100-source-screen-20261026/download_video.csv`.

| Split | Rows | Unique video IDs | Duplicate IDs across splits | Invalid/nonpositive spans |
|---|---:|---:|---:|---:|
| Train | 6,489 | 6,489 | 0 | 0 |
| Validation | 2,134 | 2,134 | 0 | 0 |
| Test | 2,167 | 2,167 | 0 | 0 |
| **Total** | **10,790** | **10,790** | **0** | **0** |

The split preflight establishes unique YouTube IDs in the published CSV, not
semantic scene independence, independence from this project's previous media,
or correctness of the event annotations. Those remain unverified. CSV receipt:
294,050 bytes, SHA-256
`f1f9b36e607ed0cea04e818cf23027acd1cec4cdd9b7c118c93408e5654be735`.

The pinned Hugging Face file-tree metadata also lists 13 raw-video archive
chunks totaling 65,775,604,313 bytes (61.26 GiB) and three annotation Parquet
files totaling 555,820 bytes. These are metadata sizes only; none of those files
was fetched. The archive size is a material local-storage cost in addition to
the unresolved terms and media-rights questions.

## Task fit and representation path

The authors describe event labels with temporal boundaries, making this a much
closer natural AV candidate than transcript-only corpora. The official code
repository says the release contains annotations and extracted audio/visual
features; its README identifies VGGish audio and I3D RGB/flow visual features.
Those are not the project's pinned EfficientAT/VideoPrism representations, so
they cannot stand in for a Path A comparison without a separately justified
encoder audit. Raw-video extraction would require retrieving the source videos
from YouTube. Before any fit, define a caller-option probability target from
the event labels, retain concurrent-event semantics, and include audio-only,
video-only, aligned AV, and source-shuffled controls.

## Rights and access discrepancy

- The project site states that the UnAV-100 dataset is available for
  commercial/research use under CC BY 4.0, while explicitly saying copyright
  in the videos remains with the original owners.
- The Hugging Face dataset card at the pinned revision declares
  `CC-BY-NC-SA-4.0`.
- The official code repository is MIT-licensed; that code license does not
  resolve the conflicting dataset-card terms or grant rights in YouTube source
  recordings.

Because the primary publication surfaces give conflicting dataset terms and
the original video rights are retained, do not choose a permissive
interpretation. Treat the candidate as **REVIEW / not admitted** until the
maintainers clarify which data components carry which terms and the source
videos' reuse rights are established. No Baidu payload, annotations, features,
or videos were fetched in this screen.

## Decision

UnAV-100 is the strongest newly screened natural AV task lead so far: its
published split CSV is large and video-ID-disjoint, and its labels describe
temporally localized events. It is not yet an admissible training or quality
source because the dataset-license declarations conflict and video rights are
not resolved. Keep the R2 natural-AV gate open; do not fit, evaluate, or inspect
the test labels/media until terms, source identity overlap, and the Decision
target are resolved. No quality claim follows from this metadata-only screen.
