# R2 fresh audiovisual source screen — 2026-10-20

## Decision

No new joint audio/video source is admitted for training or checkpoint
selection. Two recent candidates were checked against the R2 admission
contract. This screen read official papers, project pages, dataset-card/file
metadata, and repository instructions only. It did not download or open
annotations, videos, audio, validation labels, or test/audit examples.

## Candidates

| Candidate | Potential fit | Gate result | Reason |
|---|---|---|---|
| WorldSense (ICLR 2026) | Strong AV coupling, 1,662 synchronized clips, 3,172 manually annotated multiple-choice questions across 26 tasks | **REVIEW / evaluation-only** | The published paper says there are no recommended train/validation splits and the dataset is designed for evaluation. Its license statements conflict: the official paper says CC BY-NC-SA 4.0 in one section and CC BY-NC in another, while the author project page and Hugging Face dataset card say CC BY 4.0. Do not train on or select against these QA/media files. Hold as a future possible audit candidate only after rights clarification and an evaluation freeze. |
| AVSBench / AVSBench-semantic | Synchronized audio/video with sounding-object masks; upstream provides train/validation/test workflows | **NOT ADMITTED / rights and access unresolved** | The official implementation repository licenses the code Apache-2.0 but does not state the dataset/media license. It says the dataset must be requested by email; the linked official download page timed out during this check. Do not treat the code license as a dataset grant or send an access request without explicit user authorization. |

## Relevance to this project

WorldSense has the most direct match to the current R2 requirement for a
predeclared task whose answer depends on both audio and video. Its evaluation-
only design and incompatible license declarations make it unsuitable for the
current train/development loop. The official paper reports a best proprietary
system accuracy of 65.1%, so this benchmark would also be a demanding external
quality audit rather than a plausible route to the 95% aspiration.

AVSBench is a candidate for a narrowly defined sound-source localization
decision, but pixel masks are not yet a typed option-ranking task, its
dataset-use terms are not documented by the code repository, and access is
controlled by an owner request. It needs a rights packet, available data
manifest, and task conversion design before any download or experiment.

HoloAssist remains the strongest already-rights-cleared action-quality source
in this branch, but all 1,466 official train sessions are allocated between
the earlier pilot (160) and the recovery selection (1,306). The earlier pilot
also used the official 207-session validation set. Its current development
validation has been repeatedly observed, so it cannot serve as a fresh check.
The official test split remains unopened. Do not start another HoloAssist fit
or re-split these used records as if they were untouched.

## Reproduction and source records

- ICLR 2026 final paper, WorldSense abstract and dataset card:
  <https://proceedings.iclr.cc/paper_files/paper/2026/hash/55dc32df563a35bf406717fb52c118d5-Abstract-Conference.html>
- ICLR 2026 final paper PDF:
  <https://proceedings.iclr.cc/paper_files/paper/2026/file/55dc32df563a35bf406717fb52c118d5-Paper-Conference.pdf>
- WorldSense author repository/project description:
  <https://github.com/JaaackHongggg/WorldSense>
- WorldSense dataset card and file listing (metadata only):
  <https://huggingface.co/datasets/honglyhly/WorldSense>
- AVSBench official repository and request instructions:
  <https://github.com/OpenNLPLab/AVSBench>
- AVSBench official data download page (fetch timed out on 2026-10-20):
  <http://www.avlbench.opennlplab.cn/download>

The WorldSense file listing exposes one `worldsense_qa.json` plus video ZIP
shards and does not show distinct train/development/test files. Only this
metadata was read; the QA JSON and ZIP contents remain unopened.

No candidate media, labels, test set, or sealed audit was loaded. No model,
checkpoint, or run was changed. The next R2 action remains an official source
screen that produces both a clear data-use grant and fresh group-disjoint
development records before a HotGPU reservation.
