# Localized Narratives × Open Images V6 delta source screen — 2026-11-13

## Decision

**Status: `REVIEW_NOT_ADMITTED`.** This is a metadata-only follow-up to the
COCO screen. Open Images V6 offers source-level author/license metadata that
could address the COCO image-attribution gap, but voice-recording reuse rights,
cross-branch exposure, annotator/image group structure, and a valid option task
have not been established. Do not fetch annotation shards, image/audio media,
build a corpus, fit a model, or score a model from this source yet.

This is a candidate image+audio grounding task. It is not a synchronized
audio+video task and cannot close the roadmap's natural AV M-2 gate.

## New evidence relative to the COCO screen

The official Localized Narratives page documents 507,444 Open Images training
annotations over 504,413 images. An annotation includes `image_id`,
`annotator_id`, timed utterances, timed pointer traces, and a relative OGG
recording path. The page states that its annotations are CC BY 4.0. It does not
separately state the rights or consent scope for the hosted OGG recordings.

The official Open Images V6 page documents image metadata fields
`OriginalLandingURL`, `License`, `AuthorProfileURL`, `Author`, and
`OriginalMD5`. This is materially more useful for per-image attribution than
the COCO metadata screened in
[`R2_LOCALIZED_NARRATIVES_SOURCE_SCREEN_20261113.md`](R2_LOCALIZED_NARRATIVES_SOURCE_SCREEN_20261113.md).
Open Images explicitly says it makes no warranty about each image's license and
that each image's license should be verified. Therefore a candidate row would
need an exact allowlisted license, an attribution target/author, and matching
source identity; dataset membership alone is not an ALLOW decision.

Primary sources:

- [Localized Narratives format, splits, and annotation license](https://google.github.io/localized-narratives/)
- [Open Images V6 download page and image metadata schema](https://storage.googleapis.com/openimages/web/download_v6.html)
- [Open Images V6 license-status caveat](https://storage.googleapis.com/openimages/web/factsfigures_v6.html)

## Bounded access measurements

Only public documentation and HTTP `HEAD` metadata were accessed. No JSONL
annotation body, image, voice recording, feature, prediction, validation/test
label, or sealed audit was opened.

| Official object | Observed metadata |
|---|---|
| Localized Narratives Open Images train shard `00000-of-00010` | 1,670,796,438 bytes; ETag `"5d89107b4349acacc90e86311c14aa6e"`; last modified 2021-01-25 |
| Localized Narratives Open Images validation JSONL | 1,139,215,790 bytes; ETag `"9ec9c385ab42329d834762011f12e183"`; last modified 2021-01-25 |
| Open Images V6 train image metadata CSV | 638,407,721 bytes; ETag `"a4ac0bcedb5c2df4d1b1230dc6b41b8f"`; last modified 2018-05-17 |

The other nine training annotation shard sizes were not checked; no total-size estimate
is asserted. This shard size makes a full annotation scan a deliberate data
acquisition decision, so resolve the voice-rights gate and source exposure
first.

## Task hypothesis and falsifiers

A possible narrow task is: given one image, a spoken segment, and a fixed set
of independently generated region candidates, select the region the segment
refers to. Captions and pointer traces would supply training labels only; they
must not appear as inference inputs or define a candidate in a way that reveals
the target. Candidate regions must be generated independently of the trace and
include visually and semantically plausible distractors.

This task is not yet shown to require both modalities. Before any fit, a
train-only metadata feasibility audit would need to establish:

- eligible image/voice rights and complete attribution for every selected row;
- source/project exposure and exact image-content overlap status;
- speaker/image connected components, with no component crossing train and
  development partitions;
- enough distinct images and annotators after connected-component grouping;
- reliable segment-to-trace alignment and a label-independent region-option
  construction;
- option-only, audio-only, and image-only controls plus region-shuffle and
  audio-swap sensitivity checks.

The current official validation speaker overlap in the COCO shard does not
prove that a new Open Images train-only grouped split is impossible. It also
does not establish that such a split exists. No labels, rows, or group counts
were inspected for Open Images in this screen. The current roadmap also records
prior Open Images image-probe and feature work, including a previously exposed
168-example probe whose predictions were insensitive to image perturbation.
The new candidate therefore has **known source-family exposure**; exact
Open-Images-ID / content overlap with those prior cohorts is not reconciled.
Those cohorts must be excluded before any candidate split is considered clean.

## Stop/go sequence

1. Reconcile this source against prior branch/PR and external-artifact exposure
   before any annotation body is acquired; exclude every previously used image
   ID and matching content from any prospective development cohort.
2. Resolve whether the publisher's CC BY 4.0 annotation statement explicitly
   covers the separate OGG voice files for model training; do not infer
   participant consent from the license statement.
3. Only if the source-policy gate passes, freeze a bounded Open Images
   TRAIN-only metadata acquisition with immutable shard and image-metadata
   revisions/hashes, then evaluate image/license/attribution and grouped-split
   support without reading official validation/test payloads.
4. Stop if any step lacks affirmative rights, clean exposure, independent
   support, or task-label integrity. No model or GPU work is justified before
   those gates pass.

## Scope and preservation

This screen used no GPU, downloaded no annotation/data payload, and changed no
existing run, corpus, checkpoint, PR, or worktree. It does not admit the source,
claim audio+image complementarity, or alter prior COCO measurements.
