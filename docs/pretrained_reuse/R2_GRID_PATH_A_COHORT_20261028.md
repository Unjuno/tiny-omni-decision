# GRID Path A fixed speaker-held-out cohort (2026-10-28)

Status: **PASS for deterministic cohort selection and direct-source media member availability. No feature fit or task-quality result yet.**

## Frozen selection

- Corrected direct-source manifests: 23,993 train / 7,998 validation; SHA-256 values are pinned in the cohort receipt.
- Speaker groups remain the existing corrected split: 24 training speakers and 8 held-out validation speakers.
- Selection: eight deterministic, code-disjoint two-slot quartets per speaker; 192 train quartets and 64 validation quartets.
- Unique utterances: 768 train and 256 validation (four query utterances per quartet).
- Validation media used in the earlier 64-quartet pilot were excluded before the new validation selection. Train/validation media overlap is 0; overlap with the prior validation pilot is 0.
- Every selected audio and video archive member exists under the direct per-speaker source bundles. All 32 speakers were checked. The aggregate audio ZIP remains excluded.
- The source split passed full-row identity and label validation before selection. Media exact-duplicate audit remains the full 63,982-payload pass recorded in the prior report.

## Receipts

- Cohort: `C:/CodexArtifacts/pretrained-reuse-r2/grid-path-a-cohort-20261028-attempt-01/cohort.json`, SHA-256 `61b70af8cb826ded36986aaa53b55ef88f3e5973542d9237df1c951086bab752`.
- Selected-media preflight: same directory, `media-preflight.json`, SHA-256 `26b06fd0de2463fcbb0a4235db9f378c85aad8ec990c6c521c4416dff8aaa24a`.
- Selection code and regression tests: `scripts/grid_path_a_data.py`, `tests/test_grid_path_a_data.py`.

## Scope and limits

This is a narrow closed-vocabulary speech-slot task using synchronized audio and video. It can compare audio-only, video-only and joint frozen-feature readouts and matched versus mismatched video, but it is not general audiovisual reasoning. The 256 validation utterances exceed the provisional 200-clip screening floor; the validation contains only eight independent speaker groups, so speaker-cluster uncertainty remains wide. GRID is internal research only pending product-rights clearance and physical cross-corpus speaker independence is unknown. No sealed audit was accessed.

The three-video native CPU preflight measured 2.4814 seconds/video after first-call compilation. Encoding these 1,024 videos is estimated at about 43 minutes plus restarts and audio extraction; this estimate will be recalibrated from live progress before a longer wait. The feature extraction and subsequent small-head fits are still pending.
