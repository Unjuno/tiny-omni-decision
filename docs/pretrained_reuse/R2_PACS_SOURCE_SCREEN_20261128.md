# PACS physical audio-visual commonsense source screen — 2026-11-28

## Disposition

**`REVIEW_NOT_ADMITTED` for product-directed training or quality selection.**
PACS is unusually relevant to lightweight multimodal Decision research, but
the author-provided dataset statement restricts use to a benchmark and says
not to train systems intended for real-world deployment. The GitHub MIT
license is the code license; it does not clear the YouTube-derived clips or
the benchmark annotations for this product path. No annotation/media payload,
checkpoint, or model was opened or used.

## Task fit and data evidence

The authors describe a physical commonsense task that compares two objects
under a natural-language criterion. The videos include visual and acoustic
evidence, and the paper reports questions where sound helps distinguish
physical properties. The paper reports 13,400 question-answer pairs, 1,526
object videos, and 1,377 unique questions. The official repository's published
baseline table reports human accuracy of 96.3% with audio and 90.5% without
audio. These results make PACS a valuable task-design reference; they do not
predict this project's model performance or establish a product-quality pass.

The paper's data statement says PACS is intended strictly as a research
benchmark and must not be used to train systems that will be deployed in the
real world. It also states that the copyright in the source video clips
belongs to YouTube and the channels that published them. The repository's MIT
`LICENSE` is therefore not evidence of a commercial/product training grant
for the data. The released annotations do not have a separately identified
product-use license in the pinned materials inspected here.

## Split and label-quality limits

The paper describes a random 70%/30% sample split. It does not establish that
all questions attached to the same video/object, repeated question templates,
or source channels remain in one split. The paper reports multiple questions
per object pair and repeated questions across different object pairs. No
official group-aware split manifest or source-channel isolation rule was
identified in the pinned README/paper materials. Cross-split video, question,
and uploader overlap therefore remain **UNKNOWN**. No annotation file was
read to try to reconstruct or improve the split.

## Decision and next gate

Do not acquire or train on PACS for this product-directed work under its
stated research-only restriction. Reconsider only if the rights holder
provides an explicit product-development training grant covering both the
annotations and the underlying media, and a group-aware split can be audited
without opening a reserved evaluation partition. The reported human result is
useful for calibrating task ambition, not permission to repurpose the data.

This screen did not change any corpus, experiment, checkpoint, or run. No
GPU, cloud compute, or paid service was used.

## Pinned evidence

- Official author repository, pinned at commit
  [`cca89114f8121f69130c71b2926225a3e9f6d36b`](https://github.com/samuelyu2002/PACS/tree/cca89114f8121f69130c71b2926225a3e9f6d36b)
- Pinned `README.md`: 3,677 bytes; SHA-256
  `bff63b7651f85d2bc05ca877bb1da6a19cd66fcdec20805866074409ad8c1f6a`
- Pinned `LICENSE`: 1,066 bytes; SHA-256
  `151002c4275673b8f4b7ab35b31424d43d852b516d1ed3fb3298e7aca369ae54`
- Author paper: <https://arxiv.org/abs/2203.11130>
- Official ECCV paper: <https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/5310_ECCV_2022_paper.php>
- Author-provided full dataset statement: <https://www.researchgate.net/publication/359390161_PACS_A_Dataset_for_Physical_Audiovisual_CommonSense_Reasoning>
- Observed 2026-11-28; public metadata/code/license and paper only.

Machine-readable record: [`pacs-source-screen-20261128.json`](../../manifests/pretrained_reuse/pacs-source-screen-20261128.json).
