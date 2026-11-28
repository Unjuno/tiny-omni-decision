# Physion++ blinded review cards — 2026-11-28

## Purpose

The original attempt-02 package contains very tall scenario contact sheets.
This helper creates a separate, easier-to-navigate copy with one randomized
matched pair per PNG, while preserving the original images and response
template. It does not read the private answer key, change labels, or change
which 75 train-only pairs were selected.

## Artifact and integrity

- New output: `C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-blinded-review-20261128-attempt-03`
- 75 individual cards, each 2048×2070 pixels; 32 frames per side.
- Card order seed: `20261129`.
- All 10 source sheets were checked against the attempt-02 report SHA-256
  values before cropping.
- The output includes only `cards/`, a blank annotation CSV, a short README,
  and hashes. No card-to-source mapping or answer key is included.
- Output manifest SHA-256:
  `93f2983cc900bcf0d89cd0959e836b262d3e19980c9383118732a17cba49a64b`
- Output template SHA-256:
  `d0b4bc70021ddd84c31fad121ccc5d0db9efd3b05b4be18dd6af1cb409c80c59`
- Total output size: 123,749,295 bytes.
- All 75 output hashes and dimensions verified; no PNG text chunks found.
- Original attempt-02 response template hash remains unchanged.

The new builder is `scripts/split_physionpp_blinded_review_cards.py`; focused
regressions are in `tests/test_split_physionpp_blinded_review_cards.py`.
Five focused tests passed and Ruff passed. A visual layout check of one card
confirmed that the review ID and both 32-frame sequences remain visible; no
judgment was recorded and no answer key was consulted.

## Interpretation

This improves review usability and protects blind handling. It does not
validate the physical labels, establish that the event is visually decidable,
or replace the still-pending independent human review. The examples are
train-only and already used during development, so they are not a blind model
benchmark.
