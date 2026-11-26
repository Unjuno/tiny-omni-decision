# Video loss-weight and sampler audit

Read-only reconstruction of the completed token experiment. No new training,
validation inference, checkpoint selection or sealed audit access.

All24 seeded permutations reproduce the stored sample-order SHA256 exactly:
`4009558c773ea4db3f90772e2446dca94f11d9b164cbdf33ec37b2c3b3bcd1c8`.
3794 examples yield59 full64-example batches plus one18-example tail per epoch:
1440 optimizer updates,1416 full batches and24 tails.

The objective assigns equal total weight per task type, then scene, then choice.
Its weights sum to one. The trainer computes each effective batch loss as
`N / B * sum(w_i * CE_i)` across its microbatches. Each microbatch uses the
same effective-batch denominator, so splitting into8-example pieces does not
multiply the intended objective by the accumulation count. Uniform random
permutation makes this an unbiased batch estimator of the weighted objective.
The final18-example batch still receives one optimizer update; it is not64 examples.

## Reconstructed variation (minimum / median / maximum)

| Quantity | Full64-example batches | 18-example tails |
|---|---|---|
| Loss weight mass `N/B * sum(w)` | 0.7188 / 0.9938 / 1.4259 | 0.7019 / 0.9786 / 1.3272 |
| Weight ESS `(sum w)^2 / sum(w^2)` | 33.66 / 37.09 / 60.69 | 9.33 / 10.75 / 17.12 |
| Largest example weight fraction | 2.84% / 5.64% / 7.56% | 10.28% / 18.82% / 24.05% |
| Correct-label weighted fraction | 26.56% / 50.43% / 77.73% | 17.00% / 48.74% / 75.16% |

Global weight ESS2183.31 of3794 examples; maximum/minimum individual weight
ratio5.818. ESS measures unequal weight concentration, not independent semantic
information. Shared questions/scenes introduce additional dependence.

## Interpretation

No accumulation-normalization bug is identified in this audit. Batch composition
and mass vary, particularly in small tails. This is a possible contributor to
optimization variability, not evidence that weighting caused the observed quality.
Gradient norms/directions and updates were not reconstructed; AdamW dynamics
cannot be deduced from weight mass alone. Do not silently normalize each batch
to its own weight sum, which would change the objective.

The64-example fit used a fixed full batch while the full-corpus run used shuffled
minibatches. Their difference is therefore not a clean generalization-only test.
Both corpus diversity and optimization noise changed. Any next comparison should
state these factors explicitly, retain the original objective, and measure train
fit plus development validation without using the sealed audit.

Full per-update records: `C:/CodexArtifacts/pretrained-reuse-r2/video-batch-weight-audit-01/result.json`.
Compact summary and source hashes: companion JSON. Original run remains unchanged.
No claim toward the four-modality95% target is made.
