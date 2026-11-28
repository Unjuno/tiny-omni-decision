# QIVD / AVQA natural audio-video source screen — 2026-11-28

## Disposition

**`REVIEW_NOT_ADMITTED` for this product-directed data path.** These sources
have useful audiovisual question-answer formats, but their inspected terms
and split evidence do not support their use for the present product-oriented
training and quality-selection goal. This was a metadata-only screen: no
agreement was accepted, and no annotations, media, models, or checkpoints were
downloaded or opened.

## Qualcomm Interactive Video Dataset (QIVD)

Qualcomm describes 2,900 short videos with human-generated questions, answers,
and answer timestamps. The page lists 13 categories, but its Audio-Visual
category has only 22 items; most examples concern visual content. The source
page says “research purposes only,” and links a binding Research Use agreement.
The agreement defines Research Use as non-profit internal research, excludes
Commercial Use, prohibits Commercial Use of both the data and resulting
models, prohibits combining the data with third-party data, treats the
licensed dataset as confidential, and says download/use constitutes
acceptance. No acceptance or download was performed. The source page does not
establish a public train/validation split or subject/video-group isolation.
This is not suitable for the current product-directed fit, even though parts
of its task design are relevant.

## Tsinghua AVQA

The official page describes manually annotated multiple-choice questions over
audio-video clips and includes a `question_relation` field (`View`, `Sound`,
`Both`), which could support modality controls. Its raw video source is
VGGSound, whose media are YouTube-origin; the AVQA page's stated permission is
limited to personal/classroom copies and requires prior permission for other
copying or redistribution. The inspected materials do not establish a
product-training grant for the media or a group-auditable split by underlying
video/source identity. No download or annotation inspection was performed.

## Decision

Do not acquire or use either source for product-directed fitting or quality
selection unless the respective rights holder provides explicit permission
for that use and an auditable video/source-group split is available. QIVD's
audio-visual category is also too small by itself to carry a meaningful
multimodal quality claim. Keep the task design as a reference only; continue
seeking rights-cleared natural data or use already-admitted data after its
observability and grouping gates are resolved.

## Pinned public evidence

- Qualcomm QIVD official page: <https://www.qualcomm.com/developer/software/qualcomm-interactive-video-dataset-qivd>
  - Retrieved 2026-11-28; 17,524 bytes; SHA-256
    `73d8a494d5edcd53ef14cbc207ac8116235f297dd5ca92bf5eac2c23242e3f0e`.
- Qualcomm Data License Agreement — Research Use:
  <https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/qvid-dataset-research-license-mar-27-2025.pdf>
  - Retrieved 2026-11-28; 166,211 bytes; SHA-256
    `3759feaceba7b5c73336388b342622d8cb5c46283d5e8f41d0662a50be200128`.
- Tsinghua AVQA official page: <https://mn.cs.tsinghua.edu.cn/avqa/>
  - Retrieved 2026-11-28; 20,446 bytes; SHA-256
    `ba761adbdef59e4c644c053e531dc9d5ebf7ec615216b8e6e56561cb6e2dd8b9`.
- AVQA permission text limits copying to personal/classroom use unless
  specific prior permission is obtained; underlying VGGSound YouTube media
  rights remain separate.

No model, data, checkpoint, or running process was changed. No GPU, cloud, or
paid compute was used.

Machine-readable record: [`qivd-avqa-rights-screen-20261128.json`](../../manifests/pretrained_reuse/qivd-avqa-rights-screen-20261128.json).
