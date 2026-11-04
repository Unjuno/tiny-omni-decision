# LATAM Egocentric Residential + IMU source screen (2026-10-28)

Status: **small research/prototype candidate; not admitted for model selection or product-quality claims. No data payload was downloaded.**

## Candidate and evidence reviewed

- Dataset card: [Humyn Labs LATAM Egocentric Residential (with IMU)](https://huggingface.co/datasets/humyn-labs/LATAM-Egocentric-Residential-IMU). The search-indexed card metadata and rendered dataset page were read; no payload was fetched.
- The card declares CC BY 4.0 and says recordings were made with participant consent for research use. It also warns that incidental bystanders and private home interiors can appear. Treat the current evidence as research-use scope; resolve any broader downstream use before use.
- The card lists 20 clips, 39 minutes total, 9 household-task labels (9 clips are Folding Clothes), four countries, and a single `train` split. It reports total repository size of 3.41 GB. It has no audio.
- Video and IMU are paired, but only 13 clips are marked `imu_alignment=clip`. The remaining 7 use a session stream shared across clips, and the card says each clip's offset into that stream cannot be recovered. Exclude those 7 from frame-synchronized experiments.
- Public metadata gives IMU session IDs but no verified participant-disjoint split or participant identities. Session grouping alone does not prove independent people or homes.
- Clip-level chore recognition does not establish that the target requires both video and IMU. A video-only or IMU-only baseline may be sufficient.

## Gates and disposition

This is useful only as a **small input-pipeline / synchronization prototype** after storage is available. It is not suitable for the current R2-M quality gate because:

1. Only 13 clips have recoverable clip-level synchronization; the released metadata has no participant-disjoint split and sparse class coverage, so independent multimodal generalization is not established.
2. The action label does not demonstrate complementary-input necessity. Any future experiment must first compare metadata-only, video-only, IMU-only, and joint controls and predeclare where both streams should resolve an error.
3. The full repository is reported as 3.41 GB, larger than current free disk. No download, feature extraction, or fitting is warranted under current storage constraints.
4. The card's research-use consent statement and privacy warning need to remain attached to the exact proposed use. Do not treat a dataset-card license as blanket participant/privacy clearance.

Do not use the mutable card row list as an immutable split or sealed audit. Before a prototype, pin the Hugging Face dataset commit and file manifest, verify the exact license/consent scope, group by underlying session and participant if author metadata allows, keep only verified clip-aligned rows, and preregister the input-necessity controls. No model run follows from this screen.

## Sources

- [Dataset card (canonical Hugging Face page)](https://huggingface.co/datasets/humyn-labs/LATAM-Egocentric-Residential-IMU)
- Search-indexed rendered copy used for this read-only screen: [dataset card copy](https://cdn-avatars.qwak.ai/datasets/humyn-labs/LATAM-Egocentric-Residential-IMU)

Screen scope: metadata/card only; no video, IMU record, or archive was downloaded or opened. No model was run and no quality score was computed.
