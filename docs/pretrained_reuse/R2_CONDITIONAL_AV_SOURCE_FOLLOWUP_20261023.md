# Conditional AV source follow-up — 2026-10-23

## Decision

The approved conditional audio/video interaction design remains the right
architecture hypothesis to test, but none of the three public-source leads below
currently satisfies the data/task gate for a new quality run. No source data was
downloaded, no labels or media were opened, and no training or GPU work was run.
The existing `ConditionalAVReadout`, its baseline, checkpoints, and prior results
were not changed.

## Source screen

| Source | Verified from official material | Why it does not admit the proposed run |
|---|---|---|
| Audio-Visually Indicated Actions (AVIA) | 378 synchronized RGB/raw-audio sequences; 9 participants, 14 actions, 3 scenarios | Access is limited to non-commercial research/education and requires an institutional request. That does not authorize the intended general product path. Action labels alone do not demonstrate which examples require both modalities; nine participants also cannot satisfy a provisional 200-independent-group quality gate. |
| M3ED | Synchronized event/RGB/gray/LiDAR/IMU/RTK sensing; CC BY-SA 4.0 | The listed sensor suite does not include audio, so it cannot train or evaluate the proposed audio/video interaction. No Decision task contract was established in this screen. |
| Multi-Modal-Wireless | CC BY 4.0 simulated CSI, LiDAR, RGB/depth, IMU and radar; 161,400 frames over four virtual towns | It has no audio stream and is designed for sensing/communications. It could be a future sensor/video task lead, but does not test AV complementarity or a product-relevant Decision task. |

Official references: [AVIA dataset and access terms](https://pavis.iit.it/audio-visually-indicated-actions-dataset), [M3ED](https://m3ed.io/), [M3ED data overview](https://m3ed.io/data_overview/), [Multi-Modal-Wireless](https://le-liang.github.io/mmw/).

## What remains required before fitting

Admit a source only after confirming intended model-training and distribution
rights, pinning an immutable release and checksums, and constructing group-disjoint
train/validation IDs. Audit timestamps from the actual records. Define a Decision
task and establish audio-sufficient, video-sufficient, and both-required strata;
include query-only, unimodal, missing/degraded, irrelevant-stream and correctly
relabeled relevant-swap controls. Freeze the evaluation IDs and arm budget before
fitting the existing C0 plus four conditional-AV ablation arms. Do not reuse the
exposed CLEVRER synthetic timing validation as a fresh selection set.

The architecture decision is therefore **continue to hold as a testable
hypothesis; no candidate is admitted yet**. This is a data/task gate, not evidence
that the interaction architecture fails. The source search can continue with
rights-cleared natural recordings or a narrowly specified application where both
audio and video are demonstrably necessary.
