# R2 TartanAir V2 video/IMU source and task screen — 2026-11-23

## Disposition

`REVIEW_NOT_ADMITTED`. TartanAir V2 is a promising synthetic source for a bounded visual-inertial pipeline diagnostic, but the dataset license is inconsistently recorded across official publisher surfaces. No data was downloaded, no task was frozen, and it is not yet suitable for an R2 quality result. The screen used public documentation/repository metadata only; no media, IMU payload, annotations, model, validation/test assets, or sealed audit were accessed.

The official documentation states that the TartanAir V2 dataset is CC BY 4.0 and the toolkit is MIT. However, the official Hugging Face data repository card at revision `0d2d145e973832742a2aaa04b7d2ebffc8d82817` declares `bsd-3-clause` in its `README.md` front matter and has no separate LICENSE file. Both terms appear permissive, but their scope over the data is unclear; do not download or train until the publisher clarifies which grant covers the payload. The docs describe synchronized camera and IMU streams, but the IMU is synthesized from the camera pose sequence (translation is differentiated twice and orientation once); realistic noise can be generated with provided code. This is useful for controlled visual-inertial engineering, but it is synthetic and the modalities are mathematically coupled. A model may solve a derived target from either stream alone, so both-input necessity must be measured rather than assumed.

## Pinned evidence

- Official TartanAir V2 docs: <https://tartanair.org/> and <https://tartanair.org/modalities.html>, inspected 2026-11-23. Root docs say dataset CC BY 4.0 / toolkit MIT. Modality docs state RGB is 640×640 at 10 Hz with synchronized cameras, and IMU is generated from interpolated camera poses and synchronized to the left camera.
- Official environment catalog: <https://tartanair.org/environments.html>, inspected 2026-11-23. The prose says 65 environments. A deterministic parse of the published HTML table found 74 distinct environment rows with 569 total listed trajectories (2–14 per row). This internal count discrepancy is unresolved. Captured page lengths/SHA-256: root 11,962 bytes / `7f77823915eef1e0d55754320352859f1b8bd29b37a7e5fdb55c598cb5989aaf`; environments 27,807 bytes / `3e9532fbb9070ec7126d6be6cad63ebee16c550bdded91f752ece5ab4c57c185`; modalities 19,770 bytes / `2b3fa291dcec421e2f985eaca5aa00e34b74befb29fc101dfa89119d659ae50c`.
- Official toolkit: <https://github.com/castacks/tartanairpy>, default branch `main`, commit `158a6844d782942110967325ca3082f50ab2bfc7` (GitHub API read 2026-11-23); repository metadata reports MIT. The toolkit revision is pinned separately from the dataset terms.
- Official Hugging Face data repository: <https://huggingface.co/datasets/theairlabcmu/tartanair2>, revision `0d2d145e973832742a2aaa04b7d2ebffc8d82817`, public and ungated, but dataset card metadata says `bsd-3-clause`; `LICENSE` was absent from the pinned root tree. This conflicts with the official documentation’s CC BY 4.0 data statement.

## Task and split fit

A plausible task family is short-horizon visual-inertial odometry: score supplied candidate motion-state/trajectory hypotheses using a synchronized RGB history and IMU history, with ground-truth pose-derived targets. The official documentation describes ground-truth trajectories and provides a trajectory-level data iterator. It does not provide an existing calibrated multi-choice Decision benchmark or establish that both modalities are necessary for a particular target.

There are two possible split scopes, with different claims:

- Grouping by environment protects against shared simulated-scene leakage, but yields only 65 environments by the prose count or 74 by the table count. This is below the provisional 200 independent-scene floor.
- Grouping by trajectory yields 569 listed clips, which could support a larger held-out-clip study, but trajectories from the same environment may share the same scene. That can test new paths in known environments; it cannot support an unseen-environment claim.

A trajectory split must also prove that each trajectory is a distinct underlying asset, that no frames or windows cross splits, and that option construction does not reveal trajectory IDs or target bins. The environment-count mismatch must be resolved before freezing a manifest. Download volume and a minimum legally compliant per-task sample remain `UNKNOWN`; no download-size preflight was run.

## Decision and next gate

Retain TartanAir V2 as a possible synthetic visual-inertial **pipeline** benchmark, not as product-quality evidence. Before any acquisition or model fitting, freeze one natural-to-the-source VIO target, options, target derivation, environment/trajectory grouping, and a factorial control design (RGB-only, IMU-only, joint, missing, shuffled, and time-offset). First verify from the pinned toolkit that the selected train and validation subsets can be listed with exact sizes and file identities without downloading payloads. If the task needs environment-disjoint generalization or a 200-group validation, this source cannot pass under the currently documented counts; seek another source or state that limitation explicitly.

No BERT experiment, GPU run, quantization, or model-quality claim follows from this screen.
