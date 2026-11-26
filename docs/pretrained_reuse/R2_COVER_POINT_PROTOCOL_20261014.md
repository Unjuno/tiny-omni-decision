# Fixed cover-point supervision comparison — 2026-10-14

**Execution outcome:** real-data smoke01 failed at mandatory frame0 label gate,
before any update. This standalone all356-label experiment will not be fitted or
relaxed. See [replacement preserving every original QA example](R2_ROLE_SUPERVISED_QA_PROTOCOL_20261014.md).

## Intent and limits

Test whether explicit source-role supervision can learn input-only covering-object
selection, a prerequisite for later hidden-item binding and conditional Video
decisions. This is a component experiment inside the <=200M prequantization Omni
objective, not a replacement objective or a product quality claim. No paid compute,
new encoder, encoder training, quantization, sealed audit, corpus rewrite or legacy
checkpoint changes. User autonomous strategy authorization permits execution.

## Fixed design before fitting

- Reuse every existing frozen TAPIR256×512 point bank for99 train/257 observed
  development clips, with existing source/media/pair hashes and split identities.
- All256 row-major16×16 geometric queries remain. Source grounded cover ID and
  its frame0 normalized bbox produce Boolean point membership **only as loss and
  scoring labels**. No GT box, ID, crop or selected point enters model forward.
- Missing/ambiguous role or frame0 bbox fails closed. No grid point in a valid
  bbox is a retained all-negative row; its top1 hit is0, never dropped.
- A small shared per-point MLP projects512→64; masked mean scene context is
  projected64→64 and added before a scalar membership logit. Point-permutation
  equivariance and padding inertness must hold. No image/world coordinate reversal.
- Two independently initialized seed17 arms, identical architecture, initialization
  and epoch sampler: `initial_only` zeros temporal columns4:128; `trajectory`
  retains all32 ordered motion/uncertainty observations. Both retain first xy,
  first uncertainty and384 initial appearance features. Later spatial appearance
  is not included in this first controlled supervised comparison.
- AdamW lr0.001, weight_decay0.01, batch8 clips,50 epochs, CPU4 threads,
  deterministic. Selection: earliest minimum validation clip/class-balanced binary
  NLL. Complete all50 epochs and preserve both best and final; no seed retry.
- Loss: per-clip half mean positive BCE plus half mean negative BCE if both
  classes exist; otherwise mean available-class BCE. Then mean clips. This prevents
  background-count dominance without altering or subsampling source points.

## Metrics and controls

Primary observed development results: top1 point-in-cover-bbox over **all257 clips**
and balanced binary NLL. Also natural-grid membership Accuracy/NLL/two-class
Brier/ECE15, positive fraction, positive recall, negative recall, per-clip counts.
Background-dominated binary Accuracy is not Omni/QA Accuracy and cannot satisfy
95–98% target. Keep no-positive clips and report attainable grid support separately.
Any mismatch is measured, not treated as label-free inference success.

Fixed train-position prior `(positive_count+1)/(train_clips+2)` is a no-input
reference. Selected models also get unchanged validation, all-zero features,
cyclic other-clip features and temporal-columns4:128-zero controls. No control
selects a checkpoint or triggers another training run. Compare arm top1 hits with
paired bootstrap over video IDs,10,000 resamples, seed17, percentile95% intervals;
official split is script×participant grouped, not proven participant-disjoint.
These are known development/checkpoint-selection results, not a blind audit.

Save config/source/module/inputs/order hashes, all epoch train/validation metrics,
best/final/untrained weights, optimizer/sampler state, per-point predictions,
timings, parameter/artifact sizes and exact best reload. Separate-process real-data
smoke precedes full training and cannot contaminate the full run's RNG state.
Preserve all attempts including failures. A negative result closes this bounded
supervised comparison; do not infer the whole model family's ceiling.

## Implementation and execution checklist

- [ ] CPU regressions: label-only bbox membership/boundaries/missing fail-closed,
  point equivariance, padding, finite gradients, balanced loss and exact reload.
- [ ] Verify all356 source labels and cached media/feature identities without
  removing examples; train/validation media and prepared-window overlap zero.
- [ ] Execute independent real-data two-update smoke and cost checks.
- [ ] Run both fixed50-epoch arms, preserving best/final and exact order equality.
- [ ] Compare, retain every result, record limits and next quality gate.
- [ ] Ruff/full CPU tests/manifests; commit/push and verify exact-head CI.
