# R2 synchronized audio-video source/license screen (2026-10-28)

Status: **no newly screened source is admitted for Path A product training.**
No dataset was downloaded, no agreement accepted, and no GPU work started.

## Qualcomm Interactive Video Dataset (QIVD)

The publisher describes 2,900 short videos, each with recorded question audio,
a human transcript, answer and answer timestamp. Its audio-visual category is
described as requiring sound for a complete answer. This looks like a
potentially relevant AV decision source, although most categories are not
both-modality tasks and the answers would need a separately validated
supplied-option conversion.

The publisher's download-linked Data License Agreement is a binding contract
accepted by downloading, installing, or using the data. It limits data and
result use to internal non-profit research, excludes commercial use, prohibits
combining the data with third-party data, and restricts disclosure of the data.
That is incompatible with the project's multi-source product candidate and
repository-level reproducibility. Do not accept the agreement or acquire the
dataset under the current objective.

Primary sources: [QIVD dataset page](https://www.qualcomm.com/developer/software/qualcomm-interactive-video-dataset-qivd),
[Data License Agreement – Research Use](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/qvid-dataset-research-license-mar-27-2025.pdf).

## Tsinghua AVQA

The official project page describes manually annotated multiple-choice
audio-visual questions and marks question relation as `View`, `Sound`, or
`Both`, making the task format relevant. It states the raw clips are sourced
from VGG-Sound and provides YouTube identifiers/timestamps rather than a
project-owned open-licensed video corpus.

The official page's use terms limit copies to personal/classroom use and
non-profit purposes; redistribution or republication requires prior specific
permission. This does not clear use of the underlying YouTube videos or derived
model outputs for the product goal. Keep the source at REVIEW; do not fetch the
annotations or videos and do not use it for model selection.

Primary sources: [AVQA project, data and license page](https://mn.cs.tsinghua.edu.cn/avqa/),
[VGG-Sound source page](https://www.robots.ox.ac.uk/~vgg/data/vggsound/).

## Decision and next gate

These sources confirm that a relevant annotation format exists, but neither
currently supplies project-approved product-training rights. They do not
justify another training run. R2-M remains open. Continue with a source whose
data, annotations and resulting model use are explicitly compatible, or
resolve rights through an authorized route before acquisition. Keep all
sealed-audit material unopened. This screen used only publisher pages and
license text; no data/media payload was fetched and no license was accepted.
