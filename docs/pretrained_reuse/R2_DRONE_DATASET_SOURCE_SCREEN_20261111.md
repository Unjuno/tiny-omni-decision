# Drone-detection dataset source and group screen — 2026-11-11

## Decision

- **Video-only M1:** `REVIEW_NOT_ADMITTED`. The pinned release has a relevant four-class video detection task and enough clip rows to justify one more provenance/group audit. It does not yet prove the project's minimum number of independent capture groups or a clean, reconstructable development split.
- **Joint audio-video M2:** `NOT_ADMITTED`. The source release does not provide an audio-to-video clip mapping, the README does not claim clip-level synchronization, and there are only 90 audio clips. Do not treat the paper's system-level sensor-fusion results as synchronized-data evidence for this project.
- No media bytes, MATLAB annotation payloads, model weights, training/validation examples, or sealed-audit data were opened. No model was run and no training or inference was performed.

## Pinned source and rights screen

- Repository: [DroneDetectionThesis/Drone-detection-dataset](https://github.com/DroneDetectionThesis/Drone-detection-dataset/tree/442d0c6750708cc221636f7a1fd31cbd8cb0aeb8), commit `442d0c6750708cc221636f7a1fd31cbd8cb0aeb8`, tree `5a1f20041afb6eba144b872ae6bd67ba33a98283`.
- The pinned repository includes a [CC0-1.0 license](https://github.com/DroneDetectionThesis/Drone-detection-dataset/blob/442d0c6750708cc221636f7a1fd31cbd8cb0aeb8/LICENSE); its README says the dataset is free to download, use, and edit.
- The authors' [paper](https://arxiv.org/html/2007.07396) says 49 visible clips were complemented from the YouTube channel `SK678387` (11 airplane and 38 helicopter clips), while describing them as non-copyrighted material. The pinned description sheet independently has 49 rows pointing to that channel. The repository-level CC0 statement does not establish an item-specific rights chain for those third-party clips. Exclude those 49 unless their rights are independently resolved.
- No access agreement or paid service is required by the reviewed source pages. The unresolved item is rights provenance for those 49 files, not a request to accept terms.

## Metadata-only evidence

The source's pinned `Data/Video_dataset_description.xlsx` was fetched from the raw URL at the pinned commit. It is 31,475 bytes, Git blob `1ee7612f08e0285595d12362e1c7bb12dadcbb50`, SHA-256 `0544022AE732A6B088AF4B6555A18C53A9364C2C02721D38E30D73AF1C684CAB`. The local metadata copy and inspection scripts are outside Git at `D:\CodexArtifacts\pretrained-reuse-r2\drone-dataset-source-screen-20261111-attempt-01`.

- The sheet has 650 video rows: 365 infrared and 285 visible. Class counts are Airplane 133, Bird 130, Drone 271, and Helicopter 116. Distance-bin counts are Close 133, Medium 292, and Distant 225.
- Its available columns are sensor, class, number, distance bin, inter-bin number, and drone type/internet source. It has no airport, date, flight, operator, or session ID.
- 279 `(class, number)` keys occur in both IR and visible. Only 101 of those pairs have matching distance bins; 23 share the inter-bin-number field, and 13 share both. After excluding the 49 YouTube-derived rows, 231 such keys remain. The filename number is therefore not a reliable paired-recording key; none of these counts proves synchronization or independent capture groups.
- The pinned README reports 203,328 annotated frames. Frames and temporal windows remain nested within their source clips and do not increase independent-group count.
- The source lists 90 audio files; the paper reports 30 each for drone, helicopter, and background. No audio/video identity join is published in the checked README or description sheet.

## Split and task feasibility

The pinned GitHub tree contains 1,400 entries and is not truncated. No path name contains `train`, `test`, `eval`, or `split`; the description sheet likewise has no published-split column. This confirms that a split manifest is absent from the inspected revision, while the identities of the old paper split remain unknown.

The paper describes footage from three Swedish airports and reports an earlier evaluation comprising 120 IR and 120 visible clips, plus a 240-clip training set. The current repository is version 1.1 and its README says clips were added after the paper's version. The inspected description sheet does not identify those old split members or map clips to airport/flight/session. Therefore neither the published evaluation membership nor a genuinely fresh group-disjoint development set can yet be reconstructed from the checked metadata.

A possible future **video-only** task is caller-option classification over Airplane, Bird, Drone, and Helicopter, or a preregistered Drone-vs-lookalikes decision. Before any fitting, the next gate must establish (1) item-level rights for retained clips, (2) flight/session/site grouping including IR/visible correspondences, (3) the old published evaluation IDs to quarantine, and (4) at least 1,000 decisions across at least 200 independent groups in an untouched development partition. A frame-level random split is prohibited. If capture groups or the legacy evaluation cannot be reconstructed, reject this candidate for quality claims.

The paper's system-level audio and ADS-B results are historical context only. They do not replace paired clip identity, unimodal controls, or the required input-necessity test for M2.

## Reproducibility and limits

- Upstream revision and metadata-file hashes are recorded in [`drone-detection-source-screen-20261111.json`](../../manifests/pretrained_reuse/drone-detection-source-screen-20261111.json).
- Source file inventory used GitHub API metadata only; no media or `.mat` file contents were fetched.
- This screen does not prove that the dataset is unusable. It leaves a bounded video-only provenance/group audit as a possible next step and closes the current source for joint AV use.
- Hardware, training time, evaluation time, and model metrics: not applicable; no model execution occurred.

## Primary sources

- [Pinned repository README](https://github.com/DroneDetectionThesis/Drone-detection-dataset/blob/442d0c6750708cc221636f7a1fd31cbd8cb0aeb8/README.md)
- [Pinned CC0 license](https://github.com/DroneDetectionThesis/Drone-detection-dataset/blob/442d0c6750708cc221636f7a1fd31cbd8cb0aeb8/LICENSE)
- [Pinned description workbook](https://github.com/DroneDetectionThesis/Drone-detection-dataset/blob/442d0c6750708cc221636f7a1fd31cbd8cb0aeb8/Data/Video_dataset_description.xlsx)
- [Svanström et al. paper](https://arxiv.org/html/2007.07396)
