# AVQA official-source rights recheck — 2026-10-21

## Decision

Keep the original Tsinghua AVQA source `REVIEW`; do not fetch its annotation
contents or video, create a corpus, or fit a model. It is the strongest
currently identified semantic AVQA task candidate because its authors describe
manual four-choice questions and expose a `question_relation` field with
`View`, `Sound`, and `Both`. This is the source's intended modality label, not
independent evidence that both modalities are causally necessary. Rights remain
unresolved for the dataset and underlying VGGSound/YouTube clips.

## Pinned, metadata-only inspection

- Official dataset page: <https://mn.cs.tsinghua.edu.cn/avqa/>. It describes
  57,015 source videos and 57,335 manually annotated question-answer pairs. The
  example schema includes a four-choice question and `question_relation`.
- The page says the raw videos originate from VGGSound/YouTube and provides
  download instructions, but the page does not state a dataset reuse license
  covering the complete source media and annotations.
- Official code/data repository: <https://github.com/AlyssaYoung/AVQA>, pinned
  at `a5056c1d82849aa0981f50fa4c5a57bd16658c52` (commit date
  2023-08-17T04:42:56Z).
- GitHub repository metadata reports `license: null`. The pinned root tree has
  no root-level `LICENSE`, `LICENCE`, or `COPYING`; it does contain
  `HAVF/hcrn_havf/LICENSE`, which applies to that model-code subdirectory and
  does not establish dataset/media rights.
- The pinned tree lists `data/annotation/train_qa.json` (16,854,085 bytes) and
  `data/annotation/val_qa.json` (7,040,496 bytes). Only names, sizes, and Git
  tree metadata were inspected. Neither JSON file was fetched or opened.
- The source page's annotation nomenclature does not by itself define this
  project's independent split policy. If rights are later cleared, group all
  rows by underlying YouTube source/video identity; freeze training and
  validation at that identity level, and keep any official evaluation split
  out of development selection.

## Consequence for the R2 gate

AVQA remains a promising *task-design lead*, not an admitted source. The public
repo visibility and the code subdirectory's license do not grant rights to
annotation contents or linked media. Do not reinterpret `Both` as verified
necessity, and do not train on a third-party `CC-BY` mirror without a direct,
component-level provenance and rights audit. Reopen only after a publisher
license or explicit permission covers the annotations and the relevant video
media, then verify source identity and a train/validation group partition
before any extraction or fit.

No source annotations, media, model outputs, or sealed-audit material were
accessed. No model was loaded, no GPU or cloud compute was used, and no existing
corpus, artifact, or process was modified.
