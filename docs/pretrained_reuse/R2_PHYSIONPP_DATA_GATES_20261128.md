# Physion++ data gates before further model changes — 2026-11-28

## Decision

Keep the existing frozen-feature Path A result as exploratory evidence. Do not
increase model capacity or promote its score. First resolve whether the
decision label is visually observable under the review protocol and quantify
how often the input sampling captures the labeled event. These are unresolved
data/input questions, not established explanations for the model result.

## Verified corpus integrity

The preserved Physion++ TRAIN manifest has 652 records (339 `No`, 313 `Yes`),
492 source-asset groups across 10 scenarios, and 56.57 MB of verified media.
Official metadata labels match all 652 records. The fixed validation manifest
has 148 records and no shared record, source-asset, media-hash, or normalized
content/source identity with TRAIN. Manifest hashes are recorded in the
external audit reports below. This establishes split and label-metadata
integrity; it does not establish that the labels are visually recoverable.

## Temporal coverage of labeled contact evidence

The train-only contact-timeline audit covers 313 positive and 339 negative
videos. Positive contact is brief: the median fraction of frames labeled as
contact is 9.9%, and the median first-contact time is 70.3% of the clip. Under
uniform endpoint-inclusive sampling:

| Frames sampled | Positive videos with an exact labeled contact frame | With a sample within ±4 frames |
| ---: | ---: | ---: |
| 8 | 199/313 (63.6%) | 240/313 (76.7%) |
| 16 | 217/313 (69.3%) | 241/313 (77.0%) |
| 32 | 254/313 (81.2%) | 305/313 (97.4%) |
| 64 | 277/313 (88.5%) | 313/313 (100.0%) |

These counts describe label-frame coverage, not recognition accuracy. They do
not prove that a sampled frame is perceptually sufficient, nor that increasing
frame count will improve the trained scorer. In particular, the separate
IntPhys 8-vs-16 comparison did not show a reliable score gain; it must not be
used as a causal estimate for Physion++.

## Label observability and generalization evidence

The train-only data audit formed 75 exact-recorded-factor-matched `No`/`Yes`
pairs for visual review. The blinded review sheet still has 75 rows with zero
completed judgments (SHA-256
`882432f4397fcae5b2176e7b31a096e354d7342acae4c5b643e55e1e9d8b4981`). A
separate 75-card review template is also blank. Do not inspect either answer
key until reviewer responses have been frozen and hashed. Human visual
observability therefore remains UNKNOWN.

A separate train-only leave-one-scenario-out diagnostic trained a small
readout independently per fold. Video features yielded 55.67% OOF accuracy,
2.3797 NLL, and 0.9106 mean confidence; the training-prior control yielded
51.99% accuracy and 0.6967 NLL. Scenario-macro accuracy was 55.26%, with a
46.0% minimum scenario. This is exploratory TRAIN-only evidence, not an
independent validation result. It shows that the current scorer is weak and
poorly calibrated under scenario transfer; it cannot distinguish label
ambiguity, input coverage, feature limitations, or readout generalization.

## Next gate

1. Obtain completed, independently judged blinded responses; hash them before
   opening the answer key. Report agreement, accuracy against the frozen key,
   confidence, and `unclear`/occlusion rates, with pair-level uncertainty.
2. If humans cannot reliably identify the target from the sampled cards,
   revise the task or input evidence before model work. If humans can, keep the
   task and investigate temporal feature coverage with a predeclared
   train-only experiment before changing capacity.
3. Keep validation, publisher test, and sealed audit unopened for this data
   audit. Do not claim that more frames or a larger model fixes the issue
   until matched evaluation supports it.

## Provenance and limits

- TRAIN manifest SHA-256:
  `0de1bbbb1cc54000936c1771ea54e375c214fedda36734536f31a44c4f64932b`
- Validation manifest SHA-256:
  `1254f01256197f1c2e23f8fa13dc03f87789c328f32bf768300134dd6b70d8dc`
- Train-data audit: `C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-train-data-audit-20261128-attempt-06\audit-report.json`
- Temporal sampling audit: `C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-temporal-sampling-20261128-attempt-01\audit-report.json`
- Scenario-held-out diagnostic: `C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-path-a-scenario-holdout-20261128-attempt-01\run-report.json`
- Blinded responses: `C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-blind-observability-review-20261128-attempt-01\responses.csv`
- No model training, evaluation-set access, checkpoint change, or GPU work was
  performed for this follow-up.
