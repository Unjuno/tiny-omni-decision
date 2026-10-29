# EgoViz-120 raw MCAP topic/time preflight — 2026-10-29

## Decision

**Execution: PASS for the bounded source-integrity preflight. Data admission for model fitting: NOT PASSED.** Three smallest MCAPs from the frozen train side were retrieved to a new external artifact directory. Their MCAP CRCs and stream metadata were read without decoding message payload values, video frames, or action labels. Recorded camera/action timestamps match exactly, and the nearest head-IMU sample is within 3.31 ms. This establishes timestamp agreement in these three MCAPs, not label correctness, participant independence, or a decision task that needs both modalities.

The `/action` schema itself says the temporal action segmentation was predicted by Gemini from `/ego/zed_head/left/image/compressed`. These labels therefore cannot establish an IMU-required decision task; the source is not admitted for the roadmap's both-input-required video+sensor experiment using this supervision.

No model was loaded or trained and no quality score was computed. The frozen seed-17 split assignment was consulted only to select train groups; no validation or sealed-audit media or labels were opened. No other worktree, dataset, checkpoint, or running process was changed.

## Pinned source and selection

- Dataset: `humaidtech/EgoViz-120`; publisher dataset-card revision `7a84ecd3411b9f219b02f273c438f1ea12d74f52`.
- Publisher license file: repository commit `9c91b23a66fcaa1fee083b679d5993fdfff9a64e`; the file states CC BY 4.0. This records the publisher's grant, not an independent audit of every contributor's rights.
- Frozen metadata index: `index.parquet`, 870,961 bytes, SHA-256 `c83126637cc24a7ac2b3600724b50cf9c53966b9b098acc57d5534e14ade1074`.
- The pre-existing deterministic seed-17 split assigns the selected session groups `g114`, `g110`, and `g174` to train. They are three distinct `group_id`s; the publisher has not verified that these are three different people.
- Selection rule: the three smallest `mcap_bytes` records among train groups in the frozen index. This deliberately probes small files; it is not a representative sample of the full dataset.

The publisher describes the recordings as consented and face-blurred and the dataset card states that action labels are model-generated and not human-verified. We did not independently verify the privacy processing or label truth. The publisher's general company description of human review does not establish review of these released labels. The source therefore remains unadmitted for quality fitting pending label-quality evidence and a task contract.

## Acquired files and measured transfer

All downloads used the official Hugging Face bucket over HTTPS. Before each GET, HEAD returned the same size and ETag recorded in the receipt. The downloaded bytes matched the expected size; SHA-256 was calculated locally.

| Chunk | Category | Train group | Bytes | Download seconds | MiB/s | SHA-256 |
|---|---|---:|---:|---:|---:|---|
| `c59801fe4172cf9b430cd9232dda3909` | cleaning_restroom | g114 | 150,184,348 | 12.586 | 11.38 | `c86d1d69df71953b0b7fe1d61cfbc919743d1c108a2a0adc30b5c1a449e2e7d4` |
| `f4b4e1846062bdb8193786209a44e930` | managing_trash | g110 | 151,857,236 | 9.526 | 15.20 | `e5138ac1966ae323d55de4e6c1179ef82f7addff26069e06ee9b6b6c8a21c5fe` |
| `ad15950002d490a519a26e18052fc1e6` | organizing_items | g174 | 158,907,359 | 6.853 | 22.11 | `aba7edeef1f0f12b35f143ae10c68ff2648a0f85c78c08d2a5ce0fb9db8313bf` |

Total transfer was 460,948,943 bytes. Mean elapsed time was 9.655 seconds per file; aggregate throughput was 15.18 MiB/s. These three sequential transfers are only a local estimate, not a guarantee for larger files or future network conditions. C: had 10.20 GiB free before acquisition and 9.77 GiB afterward.

## MCAP structure and timestamp findings

The isolated `mcap==1.6.0`, `zstandard==0.25.0`, and `lz4==4.4.5` packages were installed only under the external attempt directory; the repository and shared Python environment were not changed. The reader validated MCAP CRCs. It recorded topic names, schema names/encodings, payload lengths, and log/publish timestamps only. It did not parse `/action` values, decode frames into images for inspection, or open the dashboard MP4. The action schema definition is 941 bytes with SHA-256 `6a2038b3d510660b12dc683c2c31182687d0cf5906fc8d7d690f721a86c51806`; its comments identify video-only Gemini label generation.

Each file contains 23 topics and about 17,565–17,569 messages over 14.97 seconds. The schema inventory includes ROS 2 `tad_msgs/msg/ActionPhase` on `/action`, `foxglove.CompressedVideo` packets on raw head/wrist camera topics, and `sensor_msgs/msg/Imu` on head and wrist IMU topics. Per-file counts were 449–450 `/action` records, 449–450 left-head compressed-video records, and 2,987–3,001 messages per IMU stream.

Using publish timestamps:

| Chunk | `/action` timestamps exactly matching a left-head video frame | Nearest left-head IMU offset: median / p95 / max |
|---|---:|---:|
| `c59801fe4172cf9b430cd9232dda3909` | 449 / 449 | 1.281 / 2.513 / 3.310 ms |
| `f4b4e1846062bdb8193786209a44e930` | 450 / 450 | 1.331 / 2.471 / 3.234 ms |
| `ad15950002d490a519a26e18052fc1e6` | 449 / 450 | 1.309 / 2.531 / 2.946 ms |

The timestamps establish close recorded stream alignment for this small sample. They cannot establish physical sensor calibration, semantic correctness of generated action labels, or that video and IMU both improve a decision over matched unimodal controls.

## Artifacts and next gate

External artifacts, all outside Git:

- Directory: `C:/CodexArtifacts/egoviz-mcap-preflight-20261029-attempt-01/`
- Raw MCAPs: listed above; not committed.
- `download-receipt.json` SHA-256: `6b52381ebdd3f4fdff3aca7ea9d1eadf57a9d3cef89d662b8d49653c1e997c43`.
- `mcap-topic-time-audit.json` SHA-256: `29691746e93559f54a512627607d7b790997c2c7012326cdc88cf8c8021abb05`.
- A compact committed manifest records file hashes, group selection, measurements, and non-claims.

Do not use these generated labels for the both-input-required fusion experiment. If considering a video-only label-transfer probe, first conduct an explicit audit against raw camera evidence and freeze a group-disjoint task. For video+sensor, require supervision whose source is not generated from video alone, then freeze question-only/video-only/IMU-only/joint controls before fitting. The current 10-GiB free-space state supports bounded samples, not the 7.25-TB corpus.

## References

- [Official EgoViz-120 bucket README and MCAP layout](https://huggingface.co/buckets/humaidtech/EgoViz-120/tree/README.md)
- [Publisher dataset card](https://huggingface.co/datasets/humaidtech/EgoViz-120)
- [Publisher license file](https://github.com/humaidtech/EgoViz-120/blob/9c91b23a66fcaa1fee083b679d5993fdfff9a64e/LICENSE)
- [MCAP Python reader documentation](https://mcap.dev/docs/python/)
