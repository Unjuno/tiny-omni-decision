# TartanAviation source and group screen — 2026-11-14

## Decision

The **vision video-to-ADS-B candidate** is `REVIEW_NOT_ADMITTED` for training or quality evaluation: conservative aircraft grouping leaves only 126 connected video components, below the project's provisional 200-group floor. This screen does **not** reject the entire TartanAviation release. Its separate ATC speech plus ADS-B corpus (677 daily archives; the README reports 3,374 hours of speech and 661 days of trajectories) was outside this audit and remains `UNSCREENED` for task labels, independent groups, and a supplied-option Decision task. No task-quality claim is made. Do not treat the video result as evidence that the multimodal Decision objective has been met.

## Objective context

The project objective remains a lightweight Decision model with 100–200M pre-quantization parameters, able to take text, image, audio, and video inputs and return calibrated probabilities in a structured JSON response. The aspiration is 95–98% per task/modality, with no benchmark gaming. This source screen only asks whether TartanAviation is a defensible, sufficiently independent candidate; it does not test the model or product objective.

## Pinned source and rights

- Repository: `castacks/TartanAviation`, source commit `077a04abac1e50953c45b72524edd79e3a21d9bd`.
- Data tag `1.1`: `d86d1d40137fa434ed4218f228a45e0b37f1760f`.
- The pinned README and Croissant metadata state CC BY 4.0 for the dataset; the README describes commercial use. The repository code separately states BSD-3-Clause. The software Zenodo record is not used as the dataset license basis.
- The README reports 550 vision videos, 3.1M images, 888,382 boxes, 239 aircraft, and 72 aircraft recurring across videos.
- Exact reviewed-source hashes and live-listing result are in `R2_TARTANAVIATION_SOURCE_GROUP_SCREEN_20261114.json`.

## What was inspected

The metadata-only screen matched the 550 rows in the pinned `vision/weather_stats.csv` to 550 listed video ZIP objects. It fetched ZIP central-directory tails and the small per-video `acft_sink.pkl` member with HTTP byte ranges. The pickle opcode stream was checked against a restricted allowlist before parsing. Only aircraft-key membership was used in memory for grouping; raw aircraft IDs and media payloads were not written. No MP4, AVI, frames, image, audio, labels archive payload, model, or evaluation split was loaded. The script and complete sanitized derived screen are retained outside Git at:

`D:\CodexArtifacts\pretrained-reuse-r2\tartanaviation-metadata-gate-20261114-attempt-01\`

The external `group_screen.json` is 591,486 bytes and contains per-video hashes and derived component membership but no raw aircraft IDs. Its SHA-256 is recorded in the compact JSON companion.

## Group result

Across all 550 videos, the audit found 239 distinct aircraft identities; 72 occur in multiple videos and one appears in 64. Unioning videos that share any aircraft yields **126 connected components**: 108 singleton-video components, 11 two-video components, 5 three-video components, one six-video component, and one 399-video component. A more conservative day-plus-aircraft grouping yields only 6 components, including one 542-video component.

This dependency structure prevents the vision subset from meeting a credible split with the current provisional group floor. The 126 count is an upper bound for independent groups under the tested identity relation, not a claim that all 126 are otherwise independent. A possible image-to-ADS-B matching task is only a hypothesis: labels, synchronized inputs, split-safe examples, and both-input necessity were not validated. The separate speech/ADS-B task utility and group structure were not assessed here. Weather was inspected only as per-video metadata for the 550 vision objects.

## Reproducibility and limits

- Requests: 2,093 HTTP Range requests; 51,910,915 response bytes; elapsed 350.72 seconds; no request errors.
- `weather_stats.csv` SHA-256: `84c5b4e8e9ae553386cb4e484bd829d0b81a033f881547bdee91ea1b9a76fbf7`.
- README SHA-256: `1f188c391e23b755a86634452a6a1579b5b9f3ef0b835e45d4e850b0b6a3a51f`.
- Croissant SHA-256: `2205ff39d5c970c29ddff799cadda353f96dfb8428b5aff82182641814001bf`.
- Audit script SHA-256: `38a41b7f8561105553e5645dcad04bbdfa520d334000bfc451195cf18727a7ac`.
- Derived full screen SHA-256: `11d696377bfc361d81b2e27ade6bbd8fc13b5c47bee70d7f7565c74df274a673`.
- No training, feature extraction, inference, validation, sealed-audit access, or GPU use occurred.

The vision subset may be reconsidered only with a task and split design that addresses aircraft-level dependence and yields enough genuinely independent groups. A separate audio+ADS-B screen may proceed from pinned text metadata and label schemas, without opening audio payloads. Neither result authorizes media acquisition or fitting by itself.
