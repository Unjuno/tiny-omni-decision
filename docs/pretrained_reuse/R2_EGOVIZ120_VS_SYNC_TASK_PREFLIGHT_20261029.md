# EgoViz synchronized wrist video/IMU correspondence preflight — 2026-10-29

## Result

**Input integrity: PASS. Task/model quality: NOT YET MEASURED.** A deterministic, group-disjoint 16-record pilot was selected from the frozen seed-17 session split. Thirteen new MCAPs were retrieved to a new external artifact directory; three previously verified files were reused by reference without copying or changing them. Every selected file has an expected-size check and SHA-256. All 16 MCAPs passed CRC validation and contained the expected wrist-left H.264 and six-axis IMU schemas. Publisher-generated action annotations were not read. No sealed audit, model fitting, or quality evaluation was performed.

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

## Runtime smoke and next gate

A single two-second development window was decoded and passed through the existing pinned official VideoPrism JAX/CPU implementation. It returned finite `[1, 2048, 768]` tokens, reduced to `[8, 768]` by spatial pooling while retaining temporal order. The first forward included compilation and took 7.893 seconds; warmed repeat forwards took 1.747 and 1.754 seconds. GPU was not used. This suggests roughly 6–7 minutes of encoder time for 224 windows before MCAP decoding and write overhead. System RAM had about 4.5 GiB free at the smoke; batch extraction must remain bounded.

Next, build the fixed train/development window table and IMU arrays, verify no group or media-hash overlap, then extract frozen features and evaluate the preregistered controls. Only after that integrity gate should a small option-probability readout be fit. The intended combined pretrained encoder size is about 152.3M parameters before the readout and any quantization. No encoder was trained or unfrozen.

## Reproduction artifacts

- Committed input/selection manifest: `manifests/pretrained_reuse/egoviz-vs-sync-task-preflight-20261029.json`.
- External raw data and receipts: `C:/CodexArtifacts/egoviz-vs-sync-20261029-attempt-01/`.
- Acquisition manifest SHA-256 `3e05a92c9365aeb69f91bae3ed1dce8258c77b6d58941ae07a9ed0f1fc95b733`.
- All-file CRC/stream report SHA-256 `91c578074c7879515ad5983e08af8a09029b8cf2e01ebb682453a1190d11f95d`.
- One-window VideoPrism CPU smoke SHA-256 `a856f6b7db5dde40a64935300c772e8f2a129b2a37adbd88c74f48620a5da910`.
- The full media files, pretrained checkpoint, and feature outputs are kept out of Git.

Publisher/source limits remain: the repository copy is described by its publisher as CC BY 4.0, but consent and blurring claims were not independently audited; publisher session IDs are not proven participant IDs. This preflight does not admit the source for a product-quality claim.
