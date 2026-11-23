# R2 public AV source refresh — 2026-11-23

## Scope

Metadata-only review of three public audio-video dataset releases after the
2026-11-22 source refresh. Sources were checked for documented modality/task
fit, license scope, media provenance, and split metadata. No dataset payload,
media, annotations, model, validation/test labels, or sealed-audit material was
downloaded or opened. No prior exposed validation was rescored.

## Candidates

| Source (pinned metadata snapshot) | Publisher description | Gate result |
|---|---|---|
| [AVRT-20K](https://huggingface.co/datasets/CVML-TueAI/AVRT-20K/tree/9cfc00e134d8b189bb2e5bb045f47837cb881ddb) — `9cfc00e134d8b189bb2e5bb045f47837cb881ddb` | Hub metadata declares CC BY 4.0 and lists one 40,182-row split. The card describes four-choice AV questions and reasoning traces; its paper says AVRT-20K was generated from a subset of Tsinghua AVQA. The card publishes annotations, not the video/audio media. | `REVIEW_NOT_ADMITTED`. The annotation card's CC BY declaration does not establish rights to the underlying AVQA/VGG-Sound source media. AVQA's publisher describes YouTube-sourced clips and limits its stated permission to personal/classroom use without profit; original video rights remain with source owners. No product-compatible media grant was established. The single published row split also does not itself prove a frozen source-group train/validation split. |
| [AV-Skills / AV-Think](https://huggingface.co/datasets/nvidia/av-skills/tree/4429e34ee80f4fe13de31b5c8def4d86c4f0daa2) — `4429e34ee80f4fe13de31b5c8def4d86c4f0daa2` | Publisher describes synchronized audio-video QA and temporal reasoning; AV-Think is explicitly designed to use both streams. The Hub card states the merged annotations are governed by the NVIDIA OneWay Noncommercial License, and original media retain upstream terms. | `NOT_ADMITTED_FOR_PRODUCT-DIRECTED_FIT`. The noncommercial license does not fit the current product-directed work, and the mixed upstream media rights are not cleared by the annotation release. No data was acquired. |
| [ANCHOR](https://huggingface.co/datasets/AnchorBenchmark/ANCHORDataset/tree/09dd505db4d620bca4b5e1a404330dd0ebf6094c) — `09dd505db4d620bca4b5e1a404330dd0ebf6094c` | Publisher describes 2,319 video records and 8,272 AVQA entries with temporal evidence. The release is annotations and retrieval metadata; it declares CC BY-NC-ND 4.0 and says upstream videos keep their own licenses. It declares only a `train` split and no train/validation/test assignment. | `NOT_ADMITTED`. Noncommercial/no-derivatives terms and unresolved upstream media rights fail the current product gate. The one split also supplies no independent group-held validation partition. No data was acquired. |

## Decision

None of these sources advances R2-M. AVRT-20K is structurally relevant, but its
annotation license cannot be treated as a license for source media. AV-Skills
and ANCHOR have explicit terms incompatible with the current product-directed
path. Do not start training, feature extraction, or model evaluation on these
releases. Continue source admission only for aligned media with compatible
component-level rights, enough independently grouped examples, and a
predeclared task whose labels support a genuinely both-input-required test.

This is a source/rights screen, not a legal opinion or a dataset-quality
assessment. It does not establish any quality or representation ceiling.

## Sources

- [AVRT-20K pinned dataset card](https://huggingface.co/datasets/CVML-TueAI/AVRT-20K/tree/9cfc00e134d8b189bb2e5bb045f47837cb881ddb)
- [AVRT paper](https://openreview.net/pdf?id=57bSlUHBfV), which identifies AVQA as its source subset.
- [Tsinghua AVQA publisher page](https://mn.cs.tsinghua.edu.cn/avqa/), including source media and stated permission scope.
- [AV-Skills pinned card](https://huggingface.co/datasets/nvidia/av-skills/tree/4429e34ee80f4fe13de31b5c8def4d86c4f0daa2)
- [ANCHOR pinned card](https://huggingface.co/datasets/AnchorBenchmark/ANCHORDataset/tree/09dd505db4d620bca4b5e1a404330dd0ebf6094c)
