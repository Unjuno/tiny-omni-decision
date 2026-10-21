# Joint audio-video source screen: YT-NTU-AVQ (2026-10-11)

## Decision

Keep YT-NTU-AVQ as a **review-only evaluation candidate**. It is a real,
synchronized audio-video collection with overall, audio-only, video-only, and
relative audio/video subjective quality labels. Its official project README
states the media are CC BY 3.0 and annotations are CC BY 4.0; the pinned Hub
card is public, ungated, and declares CC BY 3.0 for the video data. The exact
archive is 9,228,491,494 bytes with a Hub LFS SHA-256. The license evidence is
promising, but the manifest keeps `trust_status: review` pending the full
component-level asset/provenance review. The repository's current license gate
also returns `REVIEW` because CC BY 3.0 is not in its explicit permissive
allowlist; that policy was not changed. Creative Commons states CC BY 3.0
permits commercial sharing and adaptation with attribution, while warning
that other rights such as privacy/publicity may still apply. No media was
downloaded or decoded.

This dataset is not a semantic audio-video QA benchmark. The most direct
Decision adaptation would classify a clip's overall MOS against a predefined
quality threshold, then compare audio-only, video-only, and joint frozen-feature
readouts. That would measure subjective quality prediction and joint perceptual
quality only; it would not support broad AV reasoning claims. A quality
threshold, split protocol, and any readout training remain unselected pending
the R0 product-criteria decision.

## Pinned upstream evidence

| Item | Evidence |
|---|---|
| Dataset | `ntu-avqa/YT-NTU-AVQ@5d559cb82f571fcec33bdd90529f40637bdb2097`; public and ungated according to the Hub API. |
| Archive | `YT-NTU-AVQ_release.zip`, 9,228,491,494 bytes; LFS SHA-256 `c0296b62c0bf0f4a6fdda08790f7d6fea78b551c1dbbabbcaad060fb41d4541f`. |
| Annotation metadata | Official code repo `renyu12/YT-NTU-AVQ@61da60f281199e5a13706c3d40c6d4e6a2e3731f`, file `dataset/meta_info_YT-NTU-AVQ.csv`; 481,654 bytes, SHA-256 `78656856a3e2fd970cb17e7e4052e82c1f6a463a84be1e77cd644bd9bd200259`. |
| Labels | 1,620 records; complete numeric `score_all`, `score_v`, `score_a`, `av_weight`, and corresponding standard deviations. `seq_name` is unique across rows. |
| Score ranges | `score_all` 1.754–4.722 (mean 3.470); `score_v` 1.540–4.678; `score_a` 1.755–4.609; `av_weight` 35.86–67.59. These are descriptive corpus statistics, not model metrics. |
| Sampling flags | 120 records are marked `pretest`; 324 `manual_selected`; 24 have both. Their role must be established before defining a model split. |
| Identity limitation | The inspected metadata has no original YouTube source ID. Unique `seq_name` does not prove that two files were not derived from one underlying source. A split cannot yet be certified by original media identity. |
| Policy | The new candidate manifest is `manifests/candidates/yt-ntu-avq.yaml`; it is not added to any corpus catalog, and its trust status is `review`. |

## Paper-derived construction and split constraints

The pinned paper clarifies why the flags matter. It reports that 1,296 clips
were stratified from a candidate pool using predicted audio quality, predicted
video quality, and coarse audio semantics; 324 additional post-2017 YouTube CC
clips were manually added. It describes 120 pretest clips used to establish
references and qualify annotators, followed by the remaining 1,500 formal
clips. The metadata flags match those counts (120 `pretest`, 324
`manual_selected`). If this source is ever used, keep the pretest partition out
of model selection and treat the other 1,500 as the candidate pool, subject to
source-identity grouping.

Ratings are crowdsourced single-stimulus scores on 1.0–5.0 scales (0–100 for
relative AV attention); submissions were filtered by ranking agreement and
score dispersion. The paper reports strong monotonic correlations among
overall, audio-only, and video-only scores, and that visual quality tends to
dominate overall AV quality for these user-generated clips. Thus even a clean
future split would only measure an AV perceptual-quality proxy with known
selection/annotation structure. It is not an independent semantic multimodal
reasoning benchmark, and its AudioSet/VALOR-derived sampling creates potential
representation/data overlap to audit before model evaluation.

The local `audit-dataset-manifest` result is **REVIEW**, with unresolved
`license_policy_review` and `trust_status` for the media component, and
`trust_status` for annotations. This is an honest result of the existing
allowlist and source policy, not an affirmative project authorization.

## Assessment against R2/R4

- It fills a data-type gap: an explicit synchronized audio-video pair with
  human-measured joint quality and modality-specific ratings.
- It can support a bounded test of whether cached audio and video features add
  value for *perceptual quality prediction*, if row/source grouping, sampling
  flags, a threshold/task, and user-approved R0 limits are resolved.
- It does not label whether the sound source is visible, answer questions about
  events, or test temporal order. It therefore does not resolve the current
  near-chance CLEVRER semantic reasoning controls or justify Architecture B/C.
- The 9.23GB archive is too large to ingest wholesale without the approved
  storage/time budget. The metadata is small and was read in memory only; no
  model was loaded, no media was fetched, and no training or evaluation ran.
- This is one 2026 dataset source, not an untouched audit set. Any future use
  would require a group split and a separate frozen validation/audit design,
  not repeated selection against its observed quality labels.

## Sources

- [Pinned Hub dataset card](https://huggingface.co/datasets/ntu-avqa/YT-NTU-AVQ/tree/5d559cb82f571fcec33bdd90529f40637bdb2097)
- [Pinned Hub dataset metadata API](https://huggingface.co/api/datasets/ntu-avqa/YT-NTU-AVQ/tree/5d559cb82f571fcec33bdd90529f40637bdb2097?recursive=true&expand=true)
- [Official project repository at the pinned commit](https://github.com/renyu12/YT-NTU-AVQ/tree/61da60f281199e5a13706c3d40c6d4e6a2e3731f)
- [Official project README](https://github.com/renyu12/YT-NTU-AVQ/blob/61da60f281199e5a13706c3d40c6d4e6a2e3731f/README.md)
- [ICASSP 2026 paper](https://arxiv.org/abs/2602.22659)
- [Pinned paper HTML with sampling and annotation details](https://arxiv.org/html/2602.22659v1)
- [Creative Commons BY 3.0 deed](https://creativecommons.org/licenses/by/3.0/)
