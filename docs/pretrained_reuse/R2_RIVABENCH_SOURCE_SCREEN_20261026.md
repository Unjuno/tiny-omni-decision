# RivaBench source screen (2026-10-26)

## Decision

Do not admit RivaBench as product-aligned training or validation data. Its task
formats are relevant to audio-video decision making, but the public metadata
does not establish rights to use the underlying videos. No annotation JSON,
audio, or video payload was downloaded or inspected; no model run was started.

## Pinned metadata findings

- Source: `tsinghua-ee/RivaBench`, immutable Hugging Face revision
  `9c28157436b1d670f134eb9532cb4dca0dd2f894` (HEAD at screen time).
- The dataset card labels the repository `apache-2.0`, reports one `train`
  viewer split with 542 rows, and reports 6.49 GB total. Its README describes
  `Academic.json`, `StandUp.json`, and `synthdec.json`; the README's Apache
  declaration is the only license declaration found in the repository metadata.
- A metadata-only recursive tree request reports 542 files under
  `standupaudios/` (about 1.49 GB) and 451 under `standupvideos/` (about 3.58
  GB), in addition to root metadata. The API listing was paginated/truncated;
  counts for every partition and the exact mapping from rows to media were not
  established. No files from those directories were fetched.
- The pinned README says Academic media are obtained from M³AV, while StandUp
  media are provided and SynthDec contains videos with synthetic/real labels.
  It does not state asset-level rights, source URLs/IDs, commercial-use grants,
  or redistribution permissions for StandUp or SynthDec.
- The benchmark paper reports 1,912 Academic, 2,128 StandUp, and 200 SynthDec
  QA items, and states that all video sources are YouTube. It describes
  StandUp as reasoning about comedy with speech and gestures/facial expressions;
  however, source-platform provenance alone is not a license grant. The
  available Hugging Face viewer's 542 rows also do not match the paper's total,
  and this screen did not inspect payloads to resolve that discrepancy.
- Academic is independently unsuitable for product-aligned admission: the
  paper says this partition uses the M³AV test set. M³AV's official download
  instructions state non-commercial CC BY-NC-SA terms and that video copyright
  remains with original owners. No rights specific to these assets were shown.
- Split/group integrity, question-to-video mapping, source identity, and
  normalized-content overlap are UNKNOWN because annotation payloads were not
  opened. The single viewer split is not evidence of independent
  train/validation groups.

## Reproducibility record

- Immutable source revision: `9c28157436b1d670f134eb9532cb4dca0dd2f894`.
- Pinned README SHA-256: `82713a2b160a0b932f8d849e3bc8129ed778bddd2a93d8923b051f27471ede20`
  (760 bytes).
- The Hugging Face dataset API reports 7,892,608,492 bytes stored at the pinned
  revision. The UI rounds this to 6.49 GB; the differing units/metadata views
  were not reconciled.
- Only the small README, dataset metadata/API listing, and HTTP HEAD metadata
  were read. No corpus/cache files were created locally.

## Disposition

Keep RivaBench rejected for product-aligned data admission unless the dataset
maintainers provide explicit rights covering each media asset and intended
commercial/product use, plus a pinned source/asset manifest and an auditable
group split. A future screen may separately reassess whether a rights-cleared
subset has enough independent groups and a task contract proving both audio and
video are needed. Do not infer these conditions from the repository's Apache
card tag or from the benchmark's task description.

## Official sources

- [Pinned Hugging Face dataset card](https://huggingface.co/datasets/tsinghua-ee/RivaBench/tree/9c28157436b1d670f134eb9532cb4dca0dd2f894)
- [Pinned README](https://huggingface.co/datasets/tsinghua-ee/RivaBench/blob/9c28157436b1d670f134eb9532cb4dca0dd2f894/README.md)
- [video-SALMONN-o1 paper](https://arxiv.org/abs/2502.11775)
- [M³AV official repository](https://github.com/Jack-ZC8/M3AV-dataset)
- [M³AV download/license instructions](https://raw.githubusercontent.com/Jack-ZC8/M3AV-dataset/main/download/README.md)
