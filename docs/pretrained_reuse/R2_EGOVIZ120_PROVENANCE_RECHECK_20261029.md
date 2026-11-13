# EgoViz-120 publisher license and label-provenance recheck — 2026-10-29

## Purpose

Recheck the official publisher statements before deciding whether the existing
EgoViz-120 metadata cohort can advance toward a video-plus-IMU Decision task.
This is a source/rights review only: no MCAP, preview video, sensor payload,
model weights, or evaluation labels were downloaded or opened.

## Pinned source evidence

| Source | Revision / digest | Finding |
| --- | --- | --- |
| Publisher dataset-card repository | Hugging Face revision `7a84ecd3411b9f219b02f273c438f1ea12d74f52`; raw `README.md` SHA-256 `505ce8b885dd535c8a0527a08ccf9d24f0b7ee315e019fadd2cba6b1ee6ecba1` | Dataset-specific page says action labels are model-generated and not human-verified. |
| Publisher license repository | GitHub `main` commit `9c91b23a66fcaa1fee083b679d5993fdfff9a64e`; `LICENSE` blob `0a0e759c0faad246f2964e86841d705ade7249b7`; raw response SHA-256 `fc3180a6abaafc3d5ce5ee715ac96c6393d6b73c64aadcbfba3f21bae5ca33de` | License file states CC BY 4.0 and expressly permits commercial sharing/adaptation with appropriate credit. |
| Publisher company website | Retrieved 2026-10-29 05:03 JST; response SHA-256 `fa085ae6c3c0e5a451a7ae9e6425d456fae9db74438a0d57793acb2555c7ff64` | General service description says human review at every gate and random-clip re-audits. This is not specific evidence about the released EgoViz-120 action labels. |

The dataset card and general company page make different claims about label
review. Until Humaid clarifies whether human review applied to this released
dataset, retain the dataset-card limitation and do not describe its action
labels as verified. The README's privacy section separately states that
operators consented to public release and that footage was face-blurred; this
remains the publisher's statement, not an independent media audit.

## Admission and next gate

The publisher-license-text check now passes for an attribution-compliant
research/product-data use. The source is still **REVIEW / NOT ADMITTED for a
quality fit** because:

- label provenance and quality are unresolved;
- 180 `group_id`s are sessions, not verified distinct participants;
- the deterministic metadata split has only 36 development sessions, including
  categories with two or three sessions;
- no task has shown that the decision needs both raw video and IMU;
- raw synchronized payload is 7.25 TB, with the smallest MCAP about 150 MB;
- only 0.496 GiB was free on the sole local filesystem volume at the live
  resource check, below the existing 2-GiB experiment reserve.

Do not fetch an MCAP under the current storage condition. Resume only after
storage is independently restored, then begin with one bounded raw MCAP
timestamp/schema audit, a dataset-specific label-provenance clarification, and
matched video-only / IMU-only / joint task controls. Keep the existing 144/36
session-group split frozen; do not treat it as participant-disjoint or sealed.

## Frozen local metadata and split artifact recheck — 05:07 JST

The preserved Parquet index and split report were independently re-read from
their recorded external artifact paths without opening any media:

- `index.parquet`: 870,961 bytes; SHA-256
  `c83126637cc24a7ac2b3600724b50cf9c53966b9b098acc57d5534e14ade1074`.
- `metadata-audit.json`: SHA-256
  `71a1af8e0e916a47f005d2f189774d6f6bfdd652eb78b33d06d44d109b9a5e66`.
- Session split `report.json`: SHA-256
  `d2e2869dcce0a7d96cddc7dee639794a7cf253c3f922637ab9c9699cb5e6d7d2`.
- Recomputed Parquet counts match the metadata receipt: 5,392 rows and unique
  chunk IDs, 180 session groups, 12 categories, and zero sequence IDs crossing
  groups.
- Re-running the fixed SHA-256 group-order rule from the Parquet file exactly
  reproduces the stored train/development group-hash lists, category row and
  group counts, and 144/36 group split. There are zero group or sequence IDs
  crossing the split.

This confirms integrity and determinism of the metadata-only split artifacts;
it adds no evidence about participant independence, label truth, or modality
complementarity and does not admit the source for fitting.

## Scope

This recheck consumed only small public README/license/homepage responses. No
data payload, fitting, evaluation, GPU, or sealed audit was used. It does not
change the source's training-admission state or claim model quality.
