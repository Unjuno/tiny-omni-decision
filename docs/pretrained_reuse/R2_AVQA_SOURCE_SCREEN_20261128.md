# AVQA audio-video task source screen — 2026-11-28

## Disposition

**REVIEW_NOT_ADMITTED.** AVQA is a promising task-design match, but its
annotations and underlying video rights are not cleared for this
product-directed training/evaluation path. No payload, annotations, or media
were downloaded or opened. Do not construct a training run from the public
description alone.

## Why it is relevant

The official [AVQA project page](https://mn.cs.tsinghua.edu.cn/avqa/) describes
57,335 manually annotated question-answer pairs over videos sourced from
VGG-Sound. Each record includes a four-choice answer, a `question_relation`
field (`View`, `Sound`, or `Both`), and a question type. The project page
explicitly describes questions where one modality is insufficient or
ambiguous and both audio and visual cues are intended to be combined. This is
closer to the R2 M-2 both-input-required task gate than ordinary video QA.

The source design is not yet evidence that the questions actually require both
streams. Any future use would need audio-only, video-only, both-input,
shuffled-stream, and timing/identity controls on the same video-grouped IDs.
The official page distinguishes train and test in its video CSV and labels
annotation files `train_qa.json` and `val_qa.json`; it does not, in the
material reviewed here, establish source-video or original-YouTube-ID
disjointness across every question split. A valid split would need grouping by
the underlying YouTube video identity, not QA-row ID or timestamped clip ID.

## Rights and source limits

The AVQA page's license section grants only personal/classroom use without
profit or commercial advantage and says other copying, republication,
server-posting, or redistribution requires prior specific permission and/or a
fee. That is not enough to admit annotation use for this product-directed
model work without a separately documented grant.

The source clips come from VGG-Sound/YouTube. The [VGG-Sound page](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)
states that the dataset is offered under CC BY 4.0 but that copyright remains
with original video owners. It also says its download links are no longer
available from the page. The AVQA page provides YouTube IDs and timestamps for
reconstruction, but this does not itself clear the rights to each video or
authorize downloading/rehosting those clips.

## Decision and next gate

- Keep the candidate `REVIEW_NOT_ADMITTED`; do not fetch the SharePoint bundle
  or reconstruct YouTube media.
- Reconsider only after the dataset authors grant written permission suitable
  for the intended model/product work, and per-video media rights are cleared
  or a rights-cleared alternative is supplied.
- Before any fit, inspect the annotation schema and split metadata under that
  grant; group by original video ID; audit duplicate/near-duplicate questions,
  answer-option shortcuts, `Both` label consistency, and train/validation/test
  source overlap.
- If admitted, freeze the both-input subset and source-grouped train/validation
  IDs before model selection. Keep the official test unopened until candidate
  selection is frozen.

This screen rules out AVQA as an immediate dataset choice; it does not mean the
R2 M-2 data gate is solved. The current Physion++ observability and scenario
generalization findings remain separate evidence.
