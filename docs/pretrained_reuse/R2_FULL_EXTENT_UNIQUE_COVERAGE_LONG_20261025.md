# R2 full-extent × unique-coverage long comparison

## Question and scope

The existing 32,768-update unique-coverage run used center-crop visual
features. The existing full-extent run used the repeated 6,656-question pool.
This experiment adds the fourth cell of that fixed crop × coverage comparison:
full-extent visual features with the existing fresh-coverage sampler. It changes
only the sampler relative to the full-extent repeat arm. It does not test new
Video examples: the pinned fresh pool adds 2,048 unique Text, Image, and Audio
questions and zero Video questions.

The run uses the same seed17 initialization, 32,768-update cosine schedule,
8 microbatches per update, frozen encoders, 889,345 trainable readout parameters,
and 24,346 ordered validation IDs as the completed comparators. The fixed
checkpoint train probe is the same 6,656 original-pool questions for every arm;
the full 12,800-row parent-plus-added TRAIN union is scored only at the endpoint.
The existing `native-full-extent-fit-01` and
`native-coverage-long-fresh-fit-01` artifacts remain read-only.

## Implementation

`train_native_full_extent.py` now accepts a separate config path and a `fresh`
policy while retaining the original default config and `repeat` behavior. The
fresh policy is rejected unless its regenerated full-run sample order exactly
matches the pinned completed center-crop fresh arm. The shared training probe
uses the original 6,656-row parent set in corpus order, so per-step train-probe
metrics remain comparable across repeat and fresh conditions. The fixed
12,800-row endpoint union continues to report seen versus unseen rows.

The CPU-only `prepare_native_full_extent_unique_coverage.py` checks pinned input
hashes and regenerates the pool, complete 32,768-step plan, LR curve,
validation IDs, and common train probe before any model/GPU load. It writes only
to a new output directory. The new config is
`configs/pretrained_reuse/native_full_extent_unique_coverage_long_v1.json`.

## Fixed conditions

- Fresh pool and occurrence plan: `native-coverage-plan-02/fresh-pool.json` and
  `fresh-plan.json`, SHA-pinned in the config.
- Full-extent feature cache: existing immutable
  `native-full-extent-cache-01`; all 8,936 Image and 456 Video feature banks.
- Fresh center-crop comparator: `native-coverage-long-fresh-fit-01`, pinned
  result, sample order, LR plan, validation IDs, selected/final checkpoints,
  and step 8,192/32,768 predictions.
- Full-extent repeat comparator: `native-full-extent-fit-01`; validation IDs,
  common train probe, and LR curve remain identical.
- Training: 32,768 optimizer updates; scheduled validation at
  0/512/1,024/2,048/4,096/8,192/12,288/16,384/24,576/32,768; no performance
  early stop; best checkpoint remains separate from final.
- Evaluation: same fixed development validation only. No sealed audit, model
  promotion, extra seed, quantization, or encoder update.

## Current status

The test-first implementation passed its targeted tests (26 passed) and Ruff.
The independent CPU sampler check regenerated 16,384 pool occurrences with
12,800 unique questions, including 2,048 additional unique rows each for Text,
Image, and Audio, no new Video rows, and 2,048 repeated occurrences within the
pool. Two full pool cycles yield 262,144 example presentations. The sample-order
SHA-256 matches the completed center-crop fresh run exactly. The LR SHA-256 is
`14234584342acb62df7d3987fdc835ec7beec3b04f5070ae864a9eac0c6becc9`; validation
IDs SHA-256 is
`db7d0f1d09319a48a7725ed36ccb515cdcc85a409f019400cf0062be38768f30`; the common
parent-probe SHA-256 is
`d6293594fc44a9cb6d1bf251d08b1815550d555388214927cf2937a30fc76c96`.

No GPU fit has started. At the time of this record, unrelated live process
PID26920 was responsive on the RTX 3080 Laptop and occupied 16,122/16,384 MiB;
it was left untouched. Once the GPU is available, run a separate 32-step smoke,
the fixed 32,768-step fit, and exact best/final replay in new directories:

```powershell
python -u -m scripts.train_native_full_extent --phase smoke --policy fresh --config configs/pretrained_reuse/native_full_extent_unique_coverage_long_v1.json --output C:/CodexArtifacts/pretrained-reuse-r2/native-full-extent-fresh-long-smoke-01
python -u -m scripts.train_native_full_extent --phase train --policy fresh --config configs/pretrained_reuse/native_full_extent_unique_coverage_long_v1.json --output C:/CodexArtifacts/pretrained-reuse-r2/native-full-extent-fresh-long-fit-01 --reference C:/CodexArtifacts/pretrained-reuse-r2/native-full-extent-fresh-long-smoke-01 --reference-sha256 <smoke-result-sha256>
python -u -m scripts.train_native_full_extent --phase replay --policy fresh --config configs/pretrained_reuse/native_full_extent_unique_coverage_long_v1.json --output C:/CodexArtifacts/pretrained-reuse-r2/native-full-extent-fresh-long-replay-01 --reference C:/CodexArtifacts/pretrained-reuse-r2/native-full-extent-fresh-long-fit-01 --reference-sha256 <fit-result-sha256>
```

The runner itself fails closed while the GPU is occupied. This report does not
claim the model experiment is complete or that fresh coverage improves quality.
