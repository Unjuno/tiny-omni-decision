# EgoViz synchronized wrist video/IMU correspondence preflight — 2026-10-29

## Result

**Input integrity: PASS. Exploratory task quality: measured, with strong overfitting.** A deterministic, group-disjoint 16-record pilot was selected from the frozen seed-17 session split. Thirteen new MCAPs were retrieved to a new external artifact directory; three previously verified files were reused by reference without copying or changing them. Every selected file has an expected-size check and SHA-256. All 16 MCAPs passed CRC validation and contained the expected wrist-left H.264 and six-axis IMU schemas. Publisher-generated action annotations were not read. A frozen-feature readout was later fit once on train, selected with development NLL, and evaluated on the fixed development split. No sealed audit was accessed.

The proposed target is four-way synchronized interval correspondence: given eight frames from a two-second wrist-camera window, select which of four two-second left-wrist IMU windows has the same source-clock interval. The three distractors are selected deterministically from the same recording and have a start-time separation of at least three seconds (the query windows use a one-second stride). Candidate order is shuffled; source path, group, category, timestamps, and window index are metadata only and are not model inputs. This is a timestamp-derived probe, not a natural task label or evidence of general decision quality.

## Selection and limits

- Dataset: `humaidtech/EgoViz-120`; frozen index SHA-256 `c83126637cc24a7ac2b3600724b50cf9c53966b9b098acc57d5534e14ade1074`.
- Existing seed-17 split receipt SHA-256 `d2e2869dcce0a7d96cddc7dee639794a7cf253c3f922637ab9c9699cb5e6d7d2`; assignment unit is publisher `group_id` session.
- Selection: two smallest records by `(mcap_bytes, chunk_id)` for each of four categories per split; 8 unique training groups and 8 unique development groups, with zero group overlap.
- Categories: cleaning_restroom, organizing_items, preparing_food, and mopping_floor.
- Sixteen files total 3,009,402,794 bytes. Thirteen new downloads account for 2,546,407,236 bytes; the three earlier checked files were reused read-only.
- Participant identity independence remains **UNKNOWN**. The development set has only eight publisher groups and is development-only, not a sealed audit. Results cannot establish population-level performance.
- The publisher describes its action labels as generated from video; we do not use those labels for this task. Raw synchronized timestamps define the correspondence target.

## Download timing and wait policy

The first three new files took 25.171, 16.649, and 16.059 seconds (mean 19.293 seconds; aggregate 9.21 MiB/s). Applying that mean to the ten remaining files predicted 193 seconds. Their measured transfer time was 184.243 seconds, about 4.5% faster than the estimate. Across all thirteen new files, mean transfer time was 18.625 seconds per file and aggregate throughput was 10.03 MiB/s. I polled the running transfer every 30 seconds; all files passed size, linked-ETag, and SHA-256 checks. The estimate applies to these files and this connection only.

The sample and per-file timings are retained in `C:/CodexArtifacts/egoviz-vs-sync-20261029-attempt-01/download-timing-samples.json`; its SHA-256 is `d9eb491e1594a7fb0165aa8e07916c041ef3701219c5b3c2f4587334f4670fdc`.

## Frozen feature extraction and next gate

The fixed window corpus was built and independently checked: 106 train and 110 development queries, 216 unique synchronized IMU windows, and zero train/development session overlap. A second rebuild reproduced the `imu-windows.npy`, train JSONL, and development JSONL SHA-256 values exactly. The 4-way target is the matching source-clock interval; the other three choices are deterministic same-recording windows with at least 3 seconds of start-index separation. Publisher action labels were not read.

Pinned VideoPrism-B was then run on CPU only. The saved feature tensor is `[216, 8, 768]`: spatial tokens are mean-pooled while all eight temporal positions remain ordered. Extraction took 529.64 seconds, including MCAP decode and first-call JAX compilation; the first forward took 9.32 seconds and the warmed forwards were about 1.7–1.9 seconds each. JAX reported `TFRT_CPU_0`. The 4.95 MB feature file SHA-256 is `ca698686b9c17fc0e49d0a017e71ed1f7cb7906f127eae4e77b223dac3d85202`. External artifacts are under `C:/CodexArtifacts/egoviz-vs-sync-20261029-attempt-01/window-corpus-rebuild-01/` and `C:/CodexArtifacts/egoviz-vs-sync-20261029-attempt-01/videoprism-features-cpu-01/`.

MOMENT-1-small was run frozen on the RTX 3080 using the recorded Torch 2.6.0+cu124 environment. Its `[216, 6, 16, 512]` per-channel and per-patch embeddings took 19.60 seconds at batch size 1; peak allocated/reserved VRAM was 162,927,104 / 184,549,376 bytes. The 39.56 MB compressed feature file SHA-256 is `58b9c82efb4d5782c19c24a3ba5274e7d9a3896157a60dec5d991998de75b0ff`.

The preregistered exploratory CPU readout comparison has now run once with seed 17. The joint temporal scorer (81,920 trainable parameters; two linear projections to 64 dimensions, normalized frame/patch dot product) selected epoch 27 by development NLL and stopped at epoch 39 under patience 12. On the fixed 110-example development set, it reached 47.27% Accuracy, NLL 1.2773, Brier 0.6919, and ECE 0.1501. Training accuracy at the selected checkpoint was 98.11% (NLL 0.853), a 50.84-point train/development accuracy gap and clear overfitting warning. Sensor-only (513 parameters) reached 28.18%, NLL 1.3863, Brier 0.7500, ECE 0.0239; the uniform expected baseline is 25%, NLL 1.3863, Brier 0.75, ECE 0. A paired bootstrap resampling the eight development sessions 2,000 times estimated joint-minus-sensor Accuracy +19.09 points (95% percentile interval +10.00 to +29.63 points). This interval is only exploratory because there are eight publisher session groups, participant identity independence is unknown, and the model was selected on this development set.

Two fixed controls also ran without refitting. Rolling the video-query features by half a session while keeping candidate sets unchanged reduced Accuracy to 14.55% and raised NLL to 1.4929; the joint-minus-mismatched-video Accuracy difference was +32.73 points (paired session-bootstrap 95% percentile interval +21.43 to +45.46). A fixed option permutation preserved logits up to the same permutation and preserved metrics after target remapping. This indicates sensitivity to the paired temporal video input in this constructed task, but the large train/development gap and small session count limit the result. No encoder was trained or unfrozen. The 95–98% product objective is not approached or established by this probe; it is not a general multimodal decision benchmark.

## Reproduction artifacts

- Committed input/selection manifest: `manifests/pretrained_reuse/egoviz-vs-sync-task-preflight-20261029.json`.
- External raw data and receipts: `C:/CodexArtifacts/egoviz-vs-sync-20261029-attempt-01/`.
- Acquisition manifest SHA-256 `3e05a92c9365aeb69f91bae3ed1dce8258c77b6d58941ae07a9ed0f1fc95b733`.
- All-file CRC/stream report SHA-256 `91c578074c7879515ad5983e08af8a09029b8cf2e01ebb682453a1190d11f95d`.
- One-window VideoPrism CPU smoke SHA-256 `a856f6b7db5dde40a64935300c772e8f2a129b2a37adbd88c74f48620a5da910`.
- Deterministic window-corpus rebuild SHA-256 receipt `F1B5887C8713EF2D1AA8282CE1B9F9EF9CD56E0308D88ED25AD980C717E9B44C`; the three data-array/sample-list hashes match the original build.
- VideoPrism CPU feature report is external at `C:/CodexArtifacts/egoviz-vs-sync-20261029-attempt-01/videoprism-features-cpu-01/feature-report.json`; feature array SHA-256 `ca698686b9c17fc0e49d0a017e71ed1f7cb7906f127eae4e77b223dac3d85202`.
- Frozen MOMENT sensor feature report is external at `C:/CodexArtifacts/egoviz-vs-sync-20261029-attempt-01/moment-features-gpu-01/feature-report.json`; feature array SHA-256 `58b9c82efb4d5782c19c24a3ba5274e7d9a3896157a60dec5d991998de75b0ff`.
- Readout report SHA-256 `8E87F8F0218914EB271DA0B44CBADB1CDC1EDD02167634A928051612ACC82048`; development per-example prediction JSONL SHA-256 `66D32A6DCE522A95448BCF9D16CCA8ABCF52A50093B4C99FE7A52D608CFB63F9`; joint readout state SHA-256 `8AA3D018DAECE4648938228282EBCE3A24CF4EC27ED252DA52D7FA28C6473282`; sensor-only readout state SHA-256 `B832ACF39F3938E236872BF33EFA617E55A66CBDD2F059C45E248D52BF632129`. These are under `C:/CodexArtifacts/egoviz-vs-sync-20261029-attempt-01/readout-results-cpu-03/`.
- A fresh-process replay loaded the joint and sensor-only state dictionaries and recomputed all 110 development predictions. Joint and mismatched-video probabilities matched their saved JSONL exactly (maximum absolute difference 0.0); option IDs and all metrics matched. The read-only receipt SHA-256 is `64E5DBBB3CA897878184A48238109DD1C87A14E58D2DAE67E00312E1B0FBFF31` at `C:/CodexArtifacts/egoviz-vs-sync-20261029-attempt-01/readout-reload-verification-01/reload-verification.json`. Reproduction command: `python -m scripts.verify_egoviz_vs_sync_readout_reload --corpus <window-corpus-rebuild-01> --video-cache <videoprism-features-cpu-01/video-features.npz> --sensor-cache <moment-features-gpu-01/sensor-features.npz> --result-dir <readout-results-cpu-03> --output-dir <new-empty-output-directory>`.
- The reproducible builder, feature, training and reload-verification scripts are `scripts/build_egoviz_vs_sync_window_corpus.py`, `scripts/extract_egoviz_vs_sync_video_features.py`, `scripts/extract_egoviz_vs_sync_sensor_features.py`, `scripts/train_egoviz_vs_sync_readout.py`, and `scripts/verify_egoviz_vs_sync_readout_reload.py`; their large/external outputs remain outside Git.
- The full media files, pretrained checkpoint, and feature outputs are kept out of Git.

Publisher/source limits remain: the repository copy is described by its publisher as CC BY 4.0, but consent and blurring claims were not independently audited; publisher session IDs are not proven participant IDs. This preflight does not admit the source for a product-quality claim.
