# R2 synchronized audio-video source screen — 2026-11-11

## Decision

No new audio-video task was admitted. This was an official-source metadata and
terms screen only. No dataset archive, media, annotation payload, model,
validation cohort, or sealed audit was downloaded or opened. No training or
inference was run.

The project requires more than synchronized inputs: a decision target where
both streams are needed, source-group isolation, identifiable exclusion of
previously exposed evaluation identities, and a use grant compatible with the
product direction. None of the candidates below clears those gates.

## Candidates

| Candidate | Officially documented facts | R2 gate result |
|---|---|---|
| [QuerYD](https://www.robots.ox.ac.uk/~vgg/data/queryd/) | The project page describes 200+ hours, source YouTube videos, volunteer audio descriptions, and train/test splits; its downloader identifies source video IDs and separate audio-description files. | **Not admitted.** The official project page does not state a dataset-content license or grant rights to the source videos; those videos remain third-party YouTube material. The published tasks are retrieval/localization, not a predeclared option-decision target requiring both modalities. The 1,400+ narrator count is not a count of independent source-video groups. No source license, held-out IDs, or both-input necessity can be verified from the public project materials reviewed. |
| [VoxDub](https://github.com/voxdub-dataset/VoxDub) | The repository describes about 766K clips; `video_id` groups segments by source YouTube video. Its current README says annotation archive CC BY-NC 4.0 and says source media and DubBench remain under original rights-holder terms. The project page labels the dataset CC BY-NC-SA 4.0. | **Not admitted.** The non-commercial annotation terms conflict with the product-directed use, and raw media rights remain with original owners. The corpus targets dubbing/speech generation rather than an established both-input-required Decision task. A source-video key exists, but the reviewed materials do not establish a fresh, project-wide source-disjoint decision split. The differing CC BY-NC vs CC BY-NC-SA declarations are retained as an unresolved license discrepancy, not resolved in favor of use. |
| [Multi-sensor Concert Recording Dataset](https://qualinet.github.io/databases/audiovisual/multi_sensor_concert_recording_dataset_including_professional_and_user_generated_content/) | The catalog describes synchronized multi-view video, spatial audio, sensor metadata, and alignment annotations from 18 performance takes by one ensemble. It reports CC BY-NC-SA licensing. | **Not admitted.** The non-commercial/share-alike terms do not fit the product-directed path, and 18 takes from one ensemble are below the provisional 200 independent-group floor. No suitable option-decision task or held-out source split is established. |

The prior user instruction excluding UnAV-100 remains in force; it was not reopened or accessed during this screen.

## Next R2 action

Continue official-source discovery only. Before any acquisition, a candidate must
show, from source-of-truth materials: compatible per-item rights; at least
1,000 usable examples across at least 200 independently identifiable original
record groups; a way to quarantine prior project evaluations; a fixed task
whose labels require both streams; and feasible unimodal, missing-input,
content-swap, and temporal-misalignment controls. Stop at metadata screening if
any hard gate fails. If no candidate passes, R2 should remain open and no joint
fit or quality claim should be made.

## Source evidence

- QuerYD project page: <https://www.robots.ox.ac.uk/~vgg/data/queryd/>.
- QuerYD downloader: <https://github.com/oncescuandreea/QuerYD_downloader>.
- VoxDub repository and dataset terms: <https://github.com/voxdub-dataset/VoxDub>.
- VoxDub project page and copyright notice: <https://voxdub-dataset.github.io/VoxDub/>.
- Concert recording catalog entry: <https://qualinet.github.io/databases/audiovisual/multi_sensor_concert_recording_dataset_including_professional_and_user_generated_content/>.
