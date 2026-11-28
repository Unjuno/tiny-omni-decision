# Physion++ blinded human-review instructions — v1

## Purpose

This instruction version supplements the separate human review pack at
`C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-blinded-review-20261128-attempt-02`.
It corrects the color-specific wording found in an earlier AI-oriented packet.
It does not change the 75 preselected pairs, the hidden label key, the review
images, or the model/data splits. Freeze and hash this instruction file before
any reviewer starts. The existing blank `annotation-template.csv` SHA-256 is
`d0b4bc70021ddd84c31fad121ccc5d0db9efd3b05b4be18dd6af1cb409c80c59`; its
columns are `review_id`, `left_contact_observed`, `right_contact_observed`,
`uncertain_or_occluded`, `evidence_notes`, and `reviewer`.

This is a train-only data-observability review, not a model evaluation, a
relabeling exercise, or a sealed test. The reviewed examples were already used
in development and cannot support a blind model-quality claim.

## Reviewer prompt

For each clip, assess whether the designated target visibly contacts its
target zone anywhere in the displayed timeline. Object and target colors and
shapes vary; do not assume a red object or yellow target. Judge each side
independently and do not infer one side's judgment from its paired clip. The pair may
contain zero, one, or two visibly supported contact events.

Use the supplied image sequence only. Frames are ordered left-to-right, then
top-to-bottom, and are sparse samples of the timeline. Do not infer that no
contact occurred merely because no sampled frame shows it. If the target or
zone cannot be identified, contact is occluded, or the sampled sequence is
insufficient to decide, record `unclear` for that side and explain why.

Use the separate `left_contact_observed` and `right_contact_observed` fields.
Record `yes`, `no`, or `unclear` independently for each clip:

- `yes`: contact is visibly supported in that clip;
- `no`: the clip and target/zone are interpretable, but contact is not visibly
  supported in the sampled timeline;
- `unclear`: that clip cannot be judged from the visible evidence.

Set `uncertain_or_occluded` to `yes` if either side is marked `unclear`,
otherwise `no`. In `evidence_notes`, record the side, visible frame positions,
and whether target identity, target-zone identity, occlusion, or temporal
sampling limits the judgment. Fill the review ID and reviewer identity in
their corresponding columns. Do not record source IDs, scenario names, factors,
or guessed labels.

## Review and freeze protocol

1. Give each reviewer only the randomized image pack, this instruction, and a
   blank response template. Keep the private key, labels, sample IDs, and
   scenario/factor metadata unavailable.
2. Have two reviewers judge independently. Do not resolve disagreements while
   looking at the same examples.
3. Save each completed response separately and compute its SHA-256 before
   opening or comparing any answer key. Record the instruction and image-pack
   hashes with the response hashes.
4. Only after both response files are frozen may the key custodian join them
   to the official labels. Report raw reviewer agreement and counts marked
   uncertain/occluded. Any key comparison is a train-only observability
   diagnostic; it does not modify targets or enter model selection.
5. Keep disagreements and `unclear` cases. Do not discard difficult pairs,
   change labels, alter the selected 75 pairs, or use these reviewed rows as a
   new evaluation set.

## Interpretation gate

Consistent human judgments can support that the event is visible in this
sampled representation; they do not prove the dataset is free of simulator or
rendering shortcuts. Low agreement or many `unclear` cases means the data/task
needs investigation before changing encoder capacity, frame counts, or BERT
components. The review alone cannot validate a model or establish a task
ceiling.
