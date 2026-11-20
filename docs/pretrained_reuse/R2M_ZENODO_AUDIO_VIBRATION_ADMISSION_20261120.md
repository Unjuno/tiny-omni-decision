# R2-M audio + vibration source gate (2026-11-20)

## Decision

The Zenodo **Multi-modal Dataset of Human Activities of Daily Living with Ambient Audio, Vibration and Environmental Data**, record `10.5281/zenodo.7937591`, is approved for a bounded local development experiment under CC BY 4.0 with attribution. This is a source/license preflight only. No archive, sample, audio, vibration, label row, or sealed split was downloaded or opened for this decision.

The official Zenodo API reports `cc-by-4.0`; the dataset paper independently states CC-BY 4.0. The publisher README describes 25 activities, five sensor devices, synchronized sensor observations, HDF5 rows with activity, timestamp, trial and participant identifiers, and per-activity archives. The paper reports 8,615 activity instances from 14 volunteers, with the five sensor recordings joined by unique trial ID. The audio is 16 kHz; vibration is about 1.085 kHz with per-entry sample-rate metadata. Audio and vibration were synchronized using cross-correlation. The license applies to the dataset as released; it does not license redistribution of the repository's model artifacts or permit committing any dataset payload here.

Evidence: [Zenodo record](https://zenodo.org/records/7937591), [Zenodo API metadata](https://zenodo.org/api/records/7937591), [dataset README preview](https://zenodo.org/records/7937591/preview/README.md?include_deleted=0), and [peer-reviewed data descriptor](https://www.mdpi.com/2306-5729/9/12/144). The exact file sizes and publisher MD5 values are retained in the companion JSON manifest. Downloaded content must additionally be checked against those publisher hashes and locally computed SHA-256 values before parsing.

## Why this source is a useful next step

It provides paired environmental audio and a physical vibration signal within the same labeled activity instance. This is a concrete R2-M audio+sensor route and is smaller than the HARMES full-preprocessed archive (the latter is 22.0 GB and is not selected for this first pilot). The source is intentionally not treated as video evidence: its cameras were used to monitor sessions and no video is published.

Ordinary 25-way activity recognition is **not** accepted as proof that fusion is necessary. Either modality may reveal the activity, and some labels are ambiguous or sensor-location-dependent. The first candidate task is a held-out, binary cross-modal trial-match Decision:

- Inputs: one audio segment and one vibration segment, each drawn from the same fixed sensor position and source trial or from distinct trials.
- Options: `same_trial` / `different_trial`; no natural-language question or class-name input.
- Positive: both streams map to the same publisher trial ID.
- Hard negative: different trials from the same activity class, with no shared participant/trial identity. Where available, match duration and device position before pairing.
- Split unit: participant, with trial ID as an additional indivisible group. All five device captures and every derived crop/pair from a trial remain in that participant's split.
- Required controls: audio-only and vibration-only best baselines, matched versus hard-negative pair, shuffled-pair, temporal-offset, missing-stream, and contradictory-pair interventions. Pair construction and sample IDs are frozen before fitting.

This is a candidate design, not yet established as feasible. A source-format audit must first verify that the released arrays preserve enough local timing and boundaries to create reliable within-trial windows. If negative pair construction or split class coverage is inadequate, stop before training and record the limitation. Because the dataset has only 14 participant groups, report group-level uncertainty and do not generalize this pilot to broad audio/sensor competence.

## Frozen split and audit policy

Before fitting, enumerate only the participant and trial identifiers needed to build a deterministic seed-17 participant split. Freeze the exact group-to-split mapping and hash it before extracting signal features. No participant or trial may cross train, validation, calibration, and sealed audit. The default target is 8/2/2/2 participant groups (train/calibration/validation/sealed audit); if the source's actual group inventory or class coverage makes this invalid, do not repair it by splitting trials from a participant across partitions. Instead document the limitation and revise the protocol before opening model-comparison metrics.

Sealed-audit rows and signals stay outside the iterative pipeline. Candidate choice, pairing rules, checkpoint selection, and architecture are frozen on train/calibration/validation only. Run the sealed audit once after selection. The dataset's existing activity labels are not themselves sealed evaluation labels; the new grouped split and pairing protocol must be frozen before any model fit.

## Resource and artifact boundaries

Current free space observed on C: was 209.84 GiB. This is ample for the 4.5 GB source record and a separate extraction/output directory, but source data and model weights remain outside Git under `C:\CodexArtifacts\pretrained-reuse-r2`. Do not mirror or unzip into the repository. Keep only reports, source metadata, split hashes, code, and aggregated results in Git. Recheck free space before download and extraction.

## Status

`SOURCE_LICENSE_GATE=PASS`; `PAYLOAD_HASH_GATE=PENDING`; `SOURCE_FORMAT_AUDIT=PENDING`; `SPLIT_FREEZE=PENDING`; `TASK_FEASIBILITY=PENDING`; `MODEL_FIT=NOT_STARTED`; `SEALED_AUDIT=NOT_ACCESSED`.
