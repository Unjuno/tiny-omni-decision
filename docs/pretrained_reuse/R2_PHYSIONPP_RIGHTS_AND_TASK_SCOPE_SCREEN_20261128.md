# Physion++ rights and task-scope screen — 2026-11-28

## Decision

The publisher's NeurIPS 2023 dataset paper states that the Physion++ dataset
and associated code are licensed under MIT, reports no third-party IP
restrictions, and describes the videos as simulator-generated. This supports
use of Physion++ for the project's narrowly aligned physical-prediction task,
subject to the MIT license and attribution. The paper also scopes the dataset
to measuring human/model physical prediction that requires latent-property
inference. Do not treat this as general-purpose multimodal pretraining or as
permission to use unrelated source assets.

Rights evidence is favorable, but this is not a data-quality or split-integrity
pass. The existing 652-row audit covers only the already-used `fullclip-v1`
TRAIN manifest. It does not establish independent source-group splits across
the publisher's train, readout-fitting, and test partitions, visual label
observability, or product-task performance. Human review of the label-blinded
matched-clip pack is still open. No test or sealed data was opened, and no
model was trained or selected in this screen.

## Primary-source evidence

- The [official Physion++ project page](https://dingmyu.github.io/physion_v2/)
  links the publisher's training, readout, test, and human-data archives.
- The [publisher's NeurIPS 2023 paper and dataset datasheet](https://papers.neurips.cc/paper_files/paper/2023/file/d3e8011c912e651ab2a76e7935a1e464-Paper-Datasets_and_Benchmarks.pdf)
  states that videos are generated with ThreeDWorld; the dataset is self-
  contained and contains no personal data; and the dataset and associated code
  are licensed under MIT (Dataset Datasheet, sections “Composition,”
  “Collection Process,” “Uses,” and “Distribution”).
- The same paper describes a 192-example-per-property readout-fitting split
  for mapping learned representations to the binary object-contact task and a
  separate 192-example-per-property final test. It states that the data is
  intended to measure physical prediction involving latent-property inference.

The paper is the source for the license and intended-use statements; this
screen did not infer a license from the public S3 URLs. No archive, media, or
additional metadata was downloaded for this rights check.

## R2 gate status

| Gate | Status | Evidence / remaining work |
|---|---|---|
| Dataset license for the stated physical-prediction use | Supported by publisher statement | Apply MIT attribution/notice; retain the paper and source links. |
| Task alignment | Supported narrowly | Binary prediction of whether the cued objects contact matches the stated OCP task. |
| Train label-to-official-metadata mapping | Pass for existing 652-row train corpus | See [official train-label audit](R2_PHYSIONPP_OFFICIAL_LABEL_MAPPING_AUDIT_20261128.md). |
| Independent split/group integrity across all partitions | Open | Audit official train/readout/test pair and scene identities before fitting; keep all matched/related trials within one split. Do not open final test for iteration. |
| Video evidence observability | Open | Human blind review and nuisance-cue review remain incomplete. Sparse-frame index coverage is a warning, not a quality score. |
| Path A model quality | Not measured by this rights screen | Requires a frozen, permitted train/readout protocol and matched controls. |
| Product-wide multimodal quality | Open | Physion++ is video-only task evidence and cannot close text/image/audio or joint-input gates. |

## Reproducibility

This is a read-only source screen; no model, corpus, split, or checkpoint was
modified. Source statements were read from the publisher's paper and project
page on 2026-11-28. The data-integrity and temporal-coverage measurements remain
separate in the linked train-only audit reports.
