# DAVE audio-visual Decision source screen — 2026-10-26

## Scope and pinned sources

Read-only admission screen for DAVE (Diagnostic Audio Visual Evaluation) as a
possible R2-M audio/video task. No dataset metadata payloads, JSON annotations,
video, audio, or model weights were downloaded or opened. No inference, training,
test inspection, access request, or license acceptance occurred.

- Official NeurIPS 2025 Datasets and Benchmarks paper: [DAVE](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4d5f03fdb238255019826032ae7cc8e2-Abstract-Datasets_and_Benchmarks_Track.html).
- Official code repository: [`gorjanradevski/dave@096818b91692f8eef879c264b053078ddac350f1`](https://github.com/gorjanradevski/dave/tree/096818b91692f8eef879c264b053078ddac350f1), main commit dated 2025-12-04; repository license is MIT.
- Hugging Face dataset revision: [`gorjanradevski/dave@fd4d94a1d57726b3788dff3616494a70d5864ad8`](https://huggingface.co/datasets/gorjanradevski/dave/tree/fd4d94a1d57726b3788dff3616494a70d5864ad8); dataset card declares MIT and provides `epic` and `ego4d` source collections.
- The DAVE supplementary material states that the 2,426 derived samples use EPIC-KITCHENS under CC BY-NC 4.0, Ego4D under its academic/research-only license, and ESC-50 under CC BY-NC 3.0. It creates audio overlays synthetically on the source video, aligned to annotated event boundaries. The DAVE MIT declaration therefore does not independently grant broader rights in the source audio/video or ESC-50 sounds.

The local source screen checked only Git/Hugging Face commit and tree metadata and
the official paper/terms. The repository's `data/` directory contains only a
placeholder. The pinned Hugging Face file tree lists `epic.zip` (4,849,990,113
bytes) and `ego4d.zip` (12,784,164,180 bytes), total 17,634,154,293 bytes
(about 16.42 GiB), before other files. C: had 9,854,341,120 bytes free at the
local recheck. The raw archive payloads were not fetched; no storage reservation
or cleanup was performed.

For reproducibility, metadata-only HF tree receipt:
`C:\CodexArtifacts\pretrained-reuse-r2\dave-source-screen-20261026\receipt.json`
(SHA-256 `9219b127c1a5d9207d63256f841c00c73b6a580001aefe7ff1801359331267bd`).
It records 11 tree entries and 17,644,902,937 bytes total; no payload was
downloaded.

## Task and evaluation fit

The official paper describes a 2,426-example diagnostic benchmark that aims to
require both audio and video and separates audio-visual alignment from
visual-only, audio-only, temporal-ordering and related controls. The dataset
card describes two source collections (`epic`, `ego4d`), not a project-ready
train/calibration/validation/sealed-audit split. The source videos are already used for a public
benchmark with published model results. Therefore its public evaluation should
not be treated as a fresh blind audit or used for iterative model selection.

This is a good *task-design reference* for matched audio-only, video-only,
joint, shuffled and missing-input controls. Before any future admissible use,
the team would still need authorized source terms, source-video/participant
grouping, overlap against all existing project corpora, a permitted evaluation
purpose, option/probability semantics, and a new independent development cohort.

## Disposition

**REVIEW / not admitted** for Path A training, candidate selection, product
quality evaluation, or packaging. The source restrictions are incompatible with
the currently authorized product-oriented source policy, and the listed media
archives exceed current free C: capacity. The released MIT annotation/code
metadata does not override the upstream media terms. Do not download the archives
or tune on its public evaluation results. No parameter, runtime, or model-quality
claim is made.

Continue searching for a rights-cleared, group-independent natural AV task. The
separately screened VisTouch dataset remains more promising under its explicit
CC BY 4.0 dataset grant, but its native archive is not in the Git repository;
the official Baidu browser route was blocked, so no alternate route was used.

## References

- [NeurIPS 2025 DAVE paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4d5f03fdb238255019826032ae7cc8e2-Abstract-Datasets_and_Benchmarks_Track.html)
- [NeurIPS 2025 DAVE full paper and supplementary license table](https://proceedings.neurips.cc/paper_files/paper/2025/file/4d5f03fdb238255019826032ae7cc8e2-Paper-Datasets_and_Benchmarks_Track.pdf)
- [DAVE official Git repository at the pinned commit](https://github.com/gorjanradevski/dave/tree/096818b91692f8eef879c264b053078ddac350f1)
- [DAVE Hugging Face dataset at the pinned revision](https://huggingface.co/datasets/gorjanradevski/dave/tree/fd4d94a1d57726b3788dff3616494a70d5864ad8)
- [EPIC-KITCHENS official site](https://epic-kitchens.github.io/site)
- [Ego4D official access terms](https://ego4d-data.org/docs/start-here/)
