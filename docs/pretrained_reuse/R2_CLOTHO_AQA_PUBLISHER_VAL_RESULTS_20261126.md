# Clotho-AQA publisher validation — frozen audio feature transfer

## Result and scope

The pre-registered one-shot comparison completed on CPU using the existing
question-only and audio-plus-question checkpoints. The official publisher
validation cohort was frozen before scoring. No fitting, temperature
selection, checkpoint selection, or training was performed on these rows. The
official test CSV was not opened. This validation is exposed now and must not
be reused for model selection.

The licensed, gated cohort contains 244 binary questions over 61 CC0 audio
assets and 59 uploader groups. The pre-registered minimum was 200 independent
uploader groups; it was not met. Treat all estimates below as descriptive and
do not claim a quality gate or population-level result. The 244 labels include
93 cases with annotator disagreement.

## Metrics

ECE uses 15 equal-width confidence bins. Brier is the sum over the two class
probabilities (equivalently `2 * (p_yes - target)^2`). Mean confidence is the
mean probability assigned to the predicted class.

| Frozen condition | Accuracy | NLL | Brier | ECE-15 | Mean confidence | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Question only | 0.6516 | 0.6226 | 0.4334 | 0.0591 | 0.6544 | 0.6418 |
| Audio + question | 0.6598 | 0.6171 | 0.4279 | 0.0255 | 0.6538 | 0.6409 |

The paired difference (audio + question minus question only) is `+0.82` pp
Accuracy, `−0.00550` NLL, `−0.00549` Brier, and `−0.03358` ECE. A 10,000
replicate paired uploader-cluster bootstrap (seed 17, percentile 95% intervals)
gave:

| Metric delta | Point estimate | 95% interval |
|---|---:|---:|
| Accuracy | +0.0082 | [−0.0500, +0.0685] |
| NLL | −0.00550 | [−0.02717, +0.01622] |
| Brier | −0.00549 | [−0.02636, +0.01534] |
| ECE-15 | −0.03358 | [−0.07263, +0.03896] |

All four intervals include zero. Audio improves the point estimates for
Accuracy, NLL, and Brier, but the paired evidence is inconclusive. Balanced
Accuracy is essentially unchanged. The cohort is small, clustered, and
annotator disagreement is substantial, so this does not establish that the
audio representation adds reliable task signal.

## Reproducibility and boundaries

- Preregistered cohort protocol: [`R2_CLOTHO_AQA_PUBLISHER_VAL_PROTOCOL_20261126.md`](R2_CLOTHO_AQA_PUBLISHER_VAL_PROTOCOL_20261126.md).
- Manifest SHA-256: `100630d59a92f461760a83aac1bfd6684d511904a42cf6ea1c71819b0fe5fb19`.
- Publisher validation CSV SHA-256: `c69b32095d70740cee49d66dc33690d0267cf7bc821d71d834725f9d24c0a491` (publisher MD5 was also verified).
- Frozen question encoder: MiniLM revision `4ca70771034acceecb2e72475f72050fcdde4ddc`; frozen audio encoder: EfficientAT source revision `a425fdce92572e602a1d5634799bd9f1f2efa806`.
- Audio feature extraction was checked against the existing frozen Clotho cache on two fixed clips; maximum absolute feature errors were `2.39e-7` and `1.94e-7`. MiniLM parity against eight fixed existing validation rows had maximum absolute error `2.98e-7`.
- Fixed checkpoint hashes, normalization hashes, temperatures, per-row logits/probabilities/targets/votes, and feature hashes are in the external artifact at `C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-publisher-val-20261126-attempt-01/`. The artifact integrity inventory is `result/artifact-integrity.json`; no media or model weights were added to Git.
- Executed scorer SHA-256: `6c263c88b4b1f4087a46348fd8d7a23f433fe732059d2f7ef5ce44c13073710c`. The finalizer also recorded the exact CPU audio/text encoder revisions and numerical parity. The prediction JSONL hashes are `fbf945b8c38e114ec6b6e7cb9f9af88d2f674b4b1fcc97082a7c506766372492` (question only) and `de1856a2c1b746669de37323464b48a6f36aa78934ce9817cad81d7fb0f7894c` (audio + question); metrics report SHA-256 is `d338f1fe508946016f2676850a380a3d13ff92ea0ee4e8e1e83a3f0cb594a4a7`.
- The frozen checkpoints were selected on the earlier development split. The publisher validation is not blind or sealed. EfficientAT pretraining item-level overlap is unknown. These are additional limits beyond the underpowered uploader count.
- CPU inference and bootstrap took about 6.5 seconds on Python 3.11.9 / PyTorch 2.6.0+cu124. No GPU was used; no training occurred.

The result does not justify promoting the audio readout, expanding it, or
claiming general audio Decision quality. R2 remains open; prioritize a new,
rights-admitted task with enough independent source groups and a label
definition that supports an actual supplied-option decision.
