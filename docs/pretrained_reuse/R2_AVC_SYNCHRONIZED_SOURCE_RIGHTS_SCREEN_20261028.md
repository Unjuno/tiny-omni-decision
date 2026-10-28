# R2 synchronized audiovisual source and rights screen (2026-10-28)

Status: **no new source admitted; no dataset payload downloaded**. This is a metadata and license screen only. It does not authorize accepting a separate agreement or using material whose terms conflict.

## Candidates reviewed

| Source | Officially reported content | Rights/access finding | Decision |
| --- | --- | --- | --- |
| Action Verb Corpus (AVC) | 46 episodes from 12 participants, 390 take/put/push actions; synchronized audio, video, hand/arm motion, and descriptions. This is a plausible action-grounding Decision task, but descriptions may make speech alone sufficient; modality dependence must be tested with matched controls. | OFAI's official page states CC BY 4.0. The linked Zenodo record's description states CC BY-NC-SA 4.0 and lists about 590.7 MB of files. The terms conflict. No files were downloaded and no agreement was accepted. | **REVIEW / not admitted** until the publisher clarifies the governing data license in writing and a split/task audit shows enough independent participants for validation. |
| Urban Soundscapes of the World | About 130 synchronized audiovisual recordings across nine cities. | The official page says free for research and educational purposes, and says the 360-degree videos are currently unavailable. This does not establish commercial/product-compatible rights or an available video input path. | **REJECT for current product-quality training**; metadata only, no download. |
| Ego4D | Large egocentric recordings; some are synchronized with audio, meshes, gaze, and additional cameras. | Official access instructions require accepting the dataset license agreement before access. No acceptance authority exists in this task. | **REVIEW / not accessed**; no agreement accepted and no data downloaded. |

## AVC task hypothesis and required gate

If AVC's rights are resolved, a bounded development task could ask the model to choose which typed action/object/destination tuple matches an annotated episode. The candidate must be split by participant and recording before any media is opened. A preregistered task must prevent the transcript alone or the tracker annotations alone from revealing the answer. Compare audio/speech-only, video-only, motion-only, and joint frozen-feature readouts on the same examples; include shuffled cross-participant modality controls. The current record does not establish that AVC meets those conditions or that its 12 participants support a sufficiently powered independent test.

No new download, feature extraction, fitting, or sealed evaluation was performed. The synchronized-AV quality gate remains open. Existing STARSS23, synthetic-AV, and Path A results remain unchanged.

## Primary sources checked

- [OFAI Action Verb Corpus page](https://www.ofai.at/resources/avc)
- [Zenodo Action Verb Corpus record, DOI 10.5281/zenodo.5140014](https://zenodo.org/records/5140014)
- [Urban Soundscapes recordings page](https://urban-soundscapes.org/recordings/)
- [Ego4D access documentation](https://ego4d-data.org/docs/)
