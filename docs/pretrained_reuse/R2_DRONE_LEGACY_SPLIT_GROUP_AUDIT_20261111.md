# Drone Detection legacy split and group audit — 2026-11-11

## Decision

**Do not admit the Drone Detection dataset for video-only M1 or joint audio-video M2. Do not download media, fit a readout, or score a validation set from this release.**

This follow-up checks whether the prior paper's evaluation clips and independent capture groups can be reconstructed from the pinned public metadata. They cannot. The source remains a plausible narrow detection task, but it cannot currently provide a defensible fresh, group-disjoint development split.

## Scope and sources

- Dataset repository pinned at `442d0c6750708cc221636f7a1fd31cbd8cb0aeb8` (tree `5a1f20041afb6eba144b872ae6bd67ba33a98283`).
- Description workbook pinned by SHA-256 `0544022AE732A6B088AF4B6555A18C53A9364C2C02721D38E30D73AF1C684CAB`; size 31,475 bytes.
- Reviewed the official paper, *Real-Time Drone Detection and Tracking With Visible, Thermal and Acoustic Sensors*, arXiv `2007.07396v2` (2020-10-19), especially its dataset and evaluation sections: <https://arxiv.org/html/2007.07396>.
- Reviewed the existing source screen and machine-readable manifest in this branch.

No video, audio, MATLAB annotation, test payload, model weight, or sealed-audit record was accessed. No training, inference, or GPU work was performed.

## Evidence

1. The paper says the evaluation set contains 120 IR and 120 visible clips, and that 240 other videos were selected for training. It describes the evaluation sampling by class and target-distance strata, but the inspected paper does not enumerate the selected clip filenames or stable IDs.
2. The pinned workbook has 650 rows and provides sensor, class, number, distance bin, inter-bin number, and drone type/internet source. It has no airport, flight, date, operator, session, or published train/evaluation membership field.
3. The paper identifies three Swedish airports as collection locations, but does not map each workbook row to an airport or flight/session. Thus neither an independent capture-group count nor an airport-held-out split can be verified.
4. The 49 YouTube-derived clips (11 airplane and 38 helicopter) are identifiable as a source category, but the repository-level CC0 statement does not resolve item-level rights for those third-party clips. Excluding them does not recover the missing capture groups or legacy evaluation IDs.
5. The workbook has 279 matching `(class, number)` IR/visible keys (231 after excluding the YouTube rows), but only 13 also match both distance bin and inter-bin number. This does not establish that these are paired recordings; the identifiers are not sufficient to build an audiovisual or cross-spectrum grouping map.
6. The audio release has 90 clips, but neither the checked paper nor workbook provides an audio-to-video clip mapping or synchronized group IDs. M2 remains independently unadmitted.

The paper's split description and source metadata therefore do not permit the following required safeguards: quarantine of every previously evaluated clip; grouping all views from one capture; proving at least 200 independent groups; or constructing a fresh split with at least 1,000 decisions. Frame-level or random clip-level splitting would not repair these missing identities.

## Reproducibility and disposition

- Machine-readable evidence: `manifests/pretrained_reuse/drone-legacy-split-group-audit-20261111.json`.
- This is a metadata-only admission decision, not a model result and not evidence that the dataset is unusable for every purpose.
- Reopen only if the source publishes stable clip-to-airport/flight/session identities and exact legacy evaluation IDs, and item-level rights for all retained material are resolved. Otherwise select a different candidate with a documented group manifest and rights-cleared data.

## Primary references

- [Pinned dataset repository](https://github.com/DroneDetectionThesis/Drone-detection-dataset/tree/442d0c6750708cc221636f7a1fd31cbd8cb0aeb8)
- [Pinned description workbook](https://github.com/DroneDetectionThesis/Drone-detection-dataset/blob/442d0c6750708cc221636f7a1fd31cbd8cb0aeb8/Data/Video_dataset_description.xlsx)
- [Official paper, arXiv:2007.07396v2](https://arxiv.org/html/2007.07396)
