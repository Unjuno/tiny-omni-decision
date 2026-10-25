# Open Images frozen-feature input controls — 2026-10-26

## Finding

The previously reported 79.17% validation Accuracy is **not evidence that the
correct choices depend on the image**. On the same 168 validation questions and
the already selected frozen-feature scorer, all 168 top-1 predictions were
identical under each of these conditions:

1. the original aligned VideoPrism image feature;
2. an all-zero image feature; and
3. a deterministic derangement that assigns another validation image feature
   to every question.

The derangement had zero fixed points; 159 of 168 donor images came from a
different target class. Aligned, zero-image and deranged-image Accuracy were
all 79.17%. Thus the result cannot be presented as image-grounded accuracy.
The image feature changes some probabilities, but did not change any decision.
The aligned and deranged distributions were nearly identical. This result
invalidates the initial interpretation of the pilot as evidence of useful
image-feature contribution to option selection.

| Fixed scorer input | n | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Aligned image | 168 | 79.17% | 0.5609 | 0.3014 | 0.1454 | 0.6931 |
| Zero image | 168 | 79.17% | 0.6862 | 0.3543 | 0.2123 | 0.5794 |
| Deranged validation image | 168 | 79.17% | 0.5609 | 0.3013 | 0.1453 | 0.6931 |

Aligned versus zero-image top-1 agreement was 168/168, with zero correctness
changes. The zero-image condition increased NLL by 0.1253 and Brier by 0.0529;
mean absolute probability change was 0.0586. Aligned versus deranged-image
top-1 agreement was also 168/168, with zero correctness changes; mean absolute
probability change was only 0.000303. This suggests that the current scorer's
choice ranking is explained by its text/candidate input or learned task
regularities, while the image feature has at most a confidence effect in this
pilot. The control zeroes or permutes features at inference in a scorer trained
with both modalities; it is not a separately fitted text-only baseline.

An auxiliary train-only exact option-set lookup found 155 distinct option-set
signatures in training. Only 11/168 validation questions had a signature seen
in training; majority target lookup was correct on 7/11 of those. This specific
lookup does not explain the full validation result. Other candidate-set/text
regularities remain possible and have not been isolated. A separately trained
MiniLM-only readout reached 75.60% on this same exposed validation set at a
fixed final epoch; its smaller capacity and fixed endpoint make that a
diagnostic, not a matched architecture comparison. See the
[text-only control report](R2_OPENIMAGES_TEXT_ONLY_CONTROL_20261026.md).

## Frozen protocol and integrity

- No training, checkpoint selection, calibration, sealed-audit access, media
  access, or GPU use occurred.
- All conditions use the same saved best readout, frozen MiniLM candidate
  features, validation order and labels. The derangement is fixed by seed
  `20261026`; its donor permutation is a true derangement.
- The aligned replay's maximum probability delta against the original saved
  validation predictions is exactly `0.0`.
- The result metadata records source repository commit
  `971b4b3db57cfcab07901d13b0e73ef83d0d6c76`; the exact evaluator bytes are
  identified by the SHA below and included in the follow-up source commit.
- Source best readout SHA-256:
  `5e13b70b45eece770840f75db2fd5c70da3d3045897b2a35a9ddb0a623842e0c`.
- Validation feature SHA-256:
  `2b13b7b50e026aa2c8e975239784b1d4aac5ea6343a446ba0bf490ca15fe1622`.
- Validation sample-order SHA-256:
  `785c3cd1b6fb260964b615fd27843063671a379baead722ae6a7b375ef8e1636`.
- Frozen model config SHA-256:
  `3cca69b55316b2efeeb1d458025cf79a53dbc7a0e44d4c94183dba53d4ae225b`.
- Evaluator SHA-256:
  `178cd8efca0eea9707a74687f4454bfac316f963115e43a756bdfd1e08474a74`.
- Detailed metrics, per-example probabilities, donor IDs, hashes and timestamps
  are in the external read-only result at
  `C:\CodexArtifacts\pretrained-reuse-r2\openimages-videoprism-input-controls-20261026-attempt-04\`.
  `control-metadata.json` SHA-256:
  `8f597d3bbebde9d5b6967b34634cfe4c23f120174f9ab1a6c104b9a87f015e51`;
  `control-predictions.jsonl` SHA-256:
  `b51fb6b0e437dca8530ec3728fb84498490e153568404e4e7b1d18b05ecf61e2`.

These are repeated development-validation metrics from the checkpoint-selection
split, not independent or blind estimates. The original Open Images pilot also
has unresolved per-image ownership verification, only 168 questions per split,
and a 527.6 MB uncompressed model package. Preserve it as a task-construction
and input-sensitivity diagnostic only; do not promote its 79.17% as image
quality.

## Next gate

Before another natural-image quality claim, use a newly frozen task/split and
candidate construction where target labels cannot be inferred from the
candidate set alone. Predeclare zero-image, shuffled-image and best unimodal
controls. Keep task labels and split identities independent at the image level,
and keep per-image rights unresolved rows out of product-quality evidence.
Do not retune this scorer or repeat selection on the already exposed 168
validation questions.
