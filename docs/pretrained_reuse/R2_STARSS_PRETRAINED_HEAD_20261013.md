# STARSS23 pretrained semantic output comparison

Question: does retaining the frozen audio model's learned AudioSet classifier
improve Decision learning over the pooled embedding used previously?

Executed two CPU conditions, seed17, original room splits and all12 event
queries retained. No manual taxonomy mapping: STARSS domestic-sound labels do
not have a reliable one-to-one AudioSet equivalent. Compared pooled384 inputs
against all527 pretrained logits. Both use training-only standardization and
zero padding to1152, the same74,753-parameter readout,24epochs, AdamW schedule,
minimum validation-NLL checkpoint selection and independent calibration grid.
The new representation changes feature dimensionality and semantics together;
it is not an isolated test of semantic supervision alone.

| Condition | Best epoch | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pooled features | 10 | .899691 | .270390 | .151361 | .064576 |
| Pretrained logits | 6 | .891118 | .298469 | .171457 | .094438 |

The pooled result reproduces the historical audio-only comparison. Pretrained
logits worsen all four development metrics in this fixed budget. Do not adopt
the replacement or extend training on these outcomes. This is not a claim about
all pretrained classifiers or a new blind test. Binary Accuracy is affected by
class imbalance; full per-event balanced Accuracy, counts, sensitivity and
specificity remain in the companion report. Bell has no positive training cases.

Fixed training Accuracy/NLL: pooled .974510/.087099, pretrained logits
.953922/.150548. Both exhibit train/development gaps; neither qualifies the
general audio product objective. No inference about fusion benefit follows.

Total measured head reconstruction and two-run time19.38s on CPU. Encoders
remained frozen; no new GPU work or cloud cost. Original source/checkpoint and
feature hashes were checked through existing loaders. Reconstructed logits from
the cached unnormalized pooling output using the official two linear layers and
Hardswish, with dropout disabled. A separate process compared16 pooled inputs
against the official loaded classifier: max absolute difference5.72e-6, allclose
at1e-5. This verifies classifier reconstruction, not raw-audio replay.

Command: `python -m scripts.compare_starss_pretrained_head`.
Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/starss-pretrained-head-comparison-01/`.
Includes best/final per arm, histories, predictions, normalization, protocol,
validation IDs, head parity and exact executed runner. After execution only a
line-wrap lint fix was made; the protocol's runner hash references the preserved
`executed_runner.py`. No existing corpus, artifact or sealed audit was changed.
