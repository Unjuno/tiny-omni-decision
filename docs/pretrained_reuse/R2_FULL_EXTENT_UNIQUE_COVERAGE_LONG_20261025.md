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

The test-first implementation passed its targeted tests (28 passed) and Ruff.
The CPU-only pinned preflight passed at
`C:\CodexArtifacts\pretrained-reuse-r2\native-full-extent-fresh-long-preflight-02`.
Its result SHA-256 is
`346a99beb7e6bde990c38212d57fe3abe39c1b541074a2bbda71d5a6f9f73847`; source
commit `eecb2c076016f63c2f27e8f68c05eba612d4b3df`; effective config SHA-256
`7758008edcfcedee819a6ed15b1f546f865568b2bfb56cc57a42ac2202138cb5`. All five
preflight artifact hashes were rechecked against the saved files with zero
mismatches. The independent CPU sampler check regenerated 16,384 pool
occurrences with 12,800 unique questions, including 2,048 additional unique
rows each for Text, Image, and Audio, no new Video rows, and 2,048 repeated
occurrences within the pool. Two full pool cycles yield 262,144 example
presentations. The sample-order
SHA-256 matches the completed center-crop fresh run exactly. The LR SHA-256 is
`14234584342acb62df7d3987fdc835ec7beec3b04f5070ae864a9eac0c6becc9`; validation
IDs SHA-256 is
`db7d0f1d09319a48a7725ed36ccb515cdcc85a409f019400cf0062be38768f30`; the common
parent-probe SHA-256 is
`d6293594fc44a9cb6d1bf251d08b1815550d555388214927cf2937a30fc76c96`.

Preflight attempt-01 at
`C:\CodexArtifacts\pretrained-reuse-r2\native-full-extent-fresh-long-preflight-01`
is preserved but invalid: a Windows text-writer newline conversion caused the
receipt's predicted hashes to differ from the files. No model or GPU was loaded
in that attempt. The writer now writes the exact pre-hashed bytes, with a
regression test; attempt-02 is the valid receipt above. The CPU preflight pins
the full-extent cache receipt but does not inspect its bank metadata or
stream/hash all 3.3GB of visual feature arrays. During the separate GPU smoke,
before model load, the runner checks each bank's path, SHA, shape, dtype, and
finite values.

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

## Runner attempt and provenance repair — 2026-10-25

The first HotGPU smoke attempt is preserved at
`C:\CodexArtifacts\pretrained-reuse-r2\native-full-extent-fresh-long-smoke-01`.
It ended `INCOMPLETE` at step 0, before model loading or optimizer updates,
because a pinned source file in the visual-cache receipt was compared to a newer
checkout file. The receipt pins source commit
`4ca45b517328d77f2be644c625920f8cdd22624f`; the expected hash for
`scripts/extract_frozen_videoprism_features.py` is
`0318c43e94337dbbb395d400d6c6027c2d389b8789b97b973ac37d1086129c65`. Its
current checkout hash is different, but the original commit's Git blob matches
the expected SHA-256. An audit of all 1,050 `source_pins` found six checkout
mismatches, and all six matched the exact committed blobs. No file in that
external worktree was changed.

The runner now verifies changed external source files against the visual-cache
receipt's pinned commit and records each Git-blob fallback. Its local HotGPU
guard permits only the guard process and the verified Cursor GPU renderer;
unknown GPU processes and active utilization still block launch. The measured
free memory must remain below 14,000 MiB and at least 12,288 MiB. The guard
reserves enough memory to keep Orbitune's 14,000 MiB restart threshold blocked
for the whole smoke/train/replay sequence and keeps that reservation for all
three phases. The live Orbitune
process remains untouched. Tests cover the UI-process exception and rejection
of any unrecognized auxiliary PID.

Protected sequence attempt-02 is preserved in
`native-full-extent-fresh-long-guard-02.stdout.log`, its stderr log, and
`native-full-extent-fresh-long-sequence-02.json`. The guard reserved 2,418 MiB
and kept free memory at 13,744 MiB, but the child gate rejected the guard's
`-m scripts.run_with_hotgpu_guard` command form because it only recognized the
direct `.py` spelling. This happened before an output directory was created,
before model loading, and at step 0. No training or quality measurement ran.
The gate now recognizes both verified invocation forms; attempt-03 uses new
smoke, fit, and replay output directories after CI and a matching-source CPU
preflight pass.
