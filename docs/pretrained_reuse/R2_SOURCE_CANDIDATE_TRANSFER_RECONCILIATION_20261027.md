# Perception audio/visual source-candidate transfer: existing artifact reconciliation

## Important zero-video control limitation

The saved `audio_only_zero_video` control is not a viable audio-only baseline.
In the original readout, zeroing video supplies the same all-zero visual vector
to every candidate, including `none_visible`, with the same audio vector and
same scorer. The logits are therefore mathematically tied. The saved float32
outputs have candidate-probability spreads no greater than `3.28e-7` in all
138 records; 33 nonetheless select `none_visible`. Those argmax outcomes can
depend on numerical rounding and must not be interpreted as an audio-only
capability measurement. The previously reported zero-video score is preserved
as a historical numerical artifact, but must not support multimodal benefit.

This was checked only against the existing prediction file, without a model
run or audit access. The [audit result](C:/CodexArtifacts/pretrained-reuse-r2/perception-source-zero-video-ties-20261027-attempt-01/result.json)
SHA-256 is `be74fd594edc5862ff3deacd798cf5093aaf53fb88eeb8e0f883c7c58acb5384`.
Two targeted tests and Ruff passed. A future independent audio-only baseline
needs a distinct learned none-visible score and a defined symmetric rule among
the visible candidates; a matched independently trained video-only baseline is
also required. Neither source admission nor a causal claim follows.

The external fixed-checkpoint transfer outputs in `attempt-07` had already been
produced on October 26 but were not reflected in this branch's roadmap. This
reconciliation reads the saved result and prediction hashes; it does not rerun
inference, fit a model, perform checkpoint selection, or open a sealed audit.
The related source-corpus builder is retained on
`codex/r2-perception-source-validation-transfer-20261026` at `595e4c31`.

The evaluation contains 138 official-validation video groups selected from a
previously inspected 200-video cohort; it is development evidence, not blind.
Each example asks which oracle tracked object(s), or `none_visible`, corresponds
to an audio event. The checkpoint was selected on the earlier development split
and is fixed here. Oracle tracks and event timing are inputs, so this is not
end-to-end perception. Source-label validity, identity and task independence
remain material limits.

| Saved condition | Set accuracy | Set NLL | Multiclass Brier | ECE | Visible-source hit@1 |
|---|---:|---:|---:|---:|---:|
| Joint audio + video | 0.5217 | 1.4637 | 0.5461 | 0.1243 | 0.3051 |
| Option-frequency prior, no media | 0.5725 | 1.7369 | 0.5845 | 0.2572 | 0.0000 |
| Joint with audio zeroed | 0.5507 | 1.4804 | 0.5481 | 0.1551 | 0.0678 |
| Joint with mismatched audio | 0.4130 | 1.9739 | 0.7223 | 0.2366 | 0.2203 |

The no-media prior effectively selects `none_visible`, giving 100% accuracy on
the 79 none-visible examples and 0% on the 59 visible-source examples. The
joint model gets 68.35% and 30.51%, respectively. Its aggregate accuracy is
lower than the prior's point estimate, while its NLL is better. The saved paired
video-group bootstrap for prior minus joint accuracy is +5.07 percentage points
(95% percentile interval −4.35 to +13.77); for NLL it is +0.2732 (interval
+0.0392 to +0.4951). This does not establish an accuracy difference, nor does
it justify selecting models on this previously inspected cohort.

Replacing audio with a fixed mismatched recording reduced joint accuracy by
10.87 points (paired interval −19.57 to −2.17); zeroing audio changed aggregate
accuracy by +2.90 points (interval −5.07 to +11.59), but visible-source hit@1
fell. These interventions indicate possible audio dependence, not an
independently trained unimodal baseline or proof that the task requires both
inputs. All intervals use 10,000 group resamples, seed 17; they are descriptive
of this nonblind, label-selected sample. Five-fold or larger independent-group
validation and source/label admission remain unresolved. The result is below
the four-modality product target and does not justify promotion.

Receipts are under
`C:/CodexArtifacts/pretrained-reuse-r2/perception-source-candidate-validation-transfer-20261026-attempt-07/`.
The `evaluation/result.json` SHA-256 is
`f6628aac1aca4c72edca271dd777d19be9ec757f289d54ff1a7fef9bdd28ec3d`;
`diagnostic-controls/result.json` is
`79b2626fe76492c4f31b3e7f2805481d4770dc86be6a469fff0d357e8a6410aa`;
`option-only-baseline/result.json` is
`9a62344f326991936d5af694d00a5b4eb1c4eda2dad9113bc86e4bcec43c2b97`.
The joint and prior prediction hashes match their receipts. Saved paired
comparison hashes are `15e1c65c30fd526be82afb8f3a13bcf0332af19583f7a70e5d00d49b5c3d408c`
and `98c77a7316c0f7a133282002bdc3226f5623b13984ebed81585f05ef74c86f6a`.
