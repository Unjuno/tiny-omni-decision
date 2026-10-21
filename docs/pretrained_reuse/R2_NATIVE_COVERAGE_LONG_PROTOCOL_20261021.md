# Native fresh-coverage long-budget comparison

**Status: preregistered; no new fit has started.** This follows the Roadmap's
next R2 action after the 32,768-update full-extent run: test the existing
fresh-coverage TRAIN policy at a full 32,768-update budget while restoring the
original crop and feature banks. The question is whether the earlier
2,048-update fresh-coverage result underestimates the benefit or trade-off of
more unique examples at the already measured long horizon.

## Existing comparison evidence

- Repeat control: `native-long-budget-fit-01`, seed 17, 32,768 updates, selected
  step 8,192; source `8756a09f582554c0321ed5ece71846d6b19469cd`.
- Repeat receipt SHA-256:
  `645bedbe73ed508fad68c4f636a41df6c8630cd996641c44ce2a87d31550250b`.
- Repeat selected and final checkpoints remain pinned at their existing paths;
  the new runner verifies their hashes without modifying them.
- Prior fresh-coverage comparison completed 2,048 updates and improved macro
  NLL slightly, but macro Accuracy was marginally lower and Video Accuracy fell
  4.75 percentage points. Its 2,048-step cosine horizon differs from this
  32,768-step plan, so it is descriptive history, not a steps-only comparator.
- The repeat control and new fresh run use the same original crop/features,
  seed-17 readout initialization, optimizer, LR schedule, update/microbatch
  budget, validation examples, metric aggregation and checkpoint selector.
  The only experimental intervention is replacing available repeat
  presentations with the already frozen unused-TRAIN rows under the established
  per-position modality/source/task/option-count quotas. The frozen Video pool
  has no unused questions and remains unchanged.

## Frozen run conditions

- Seed 17, fresh initialization from the pinned original native readout, fresh
  AdamW state, 32,768 optimizer updates and eight one-example microbatches per
  update. The model, loss, clipping, FP32/TF32 policy and frozen feature banks
  are unchanged.
- AdamW: LR 0.001, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01,
  gradient clip 1.0. Warmup is 62 updates; cosine is defined over the full
  32,768-update horizon from the start. No performance early stopping.
- Validation checkpoints: 0, 512, 1,024, 2,048, 4,096, 8,192, 12,288,
  16,384, 24,576 and 32,768. Best uses the existing hierarchical
  task/source/modality macro NLL selector, min_delta 1e-5, earliest tie, with
  step 0 eligible. Best and final are saved separately.
- Train pool: the original 8,192-occurrence pool plus 8,192 fresh/repeat
  occurrences from the pinned plan, extended for exactly 16 cycles. Expected
  full-run consumption is 262,144 examples. The fresh pool has 12,800 unique
  questions (6,144 added), while the repeat control has 6,656. Both keep the
  same 6,656 original-pool questions as the fixed per-checkpoint TRAIN probe;
  probe curves are diagnostics, not a fixed generalization gap.
- Full validation: same 24,346 IDs/order/options/targets as the repeat control.
  The sealed audit is not loaded. TRAIN probes/unions do not select checkpoints.
- Total trainable readout parameters remain 889,345; total pre-quantization
  inference parameter count remains 196,254,874. No encoder, temporal module,
  calibration, quantization, or architecture changes.
- Existing local GPU only. Per-process guard remains 7,200 seconds, 6 GiB
  allocated VRAM, 10 GiB peak working set and 2 GiB free-disk reserve. Each
  phase has a new output directory; no existing run, feature cache, checkpoint
  or corpus is overwritten.

## Preregistered comparisons and interpretation

Primary comparison is the fresh run's validation-selected checkpoint against
the repeat control's validation-selected checkpoint. Report equal-step
comparisons at 8,192 and 32,768, plus the fixed-step curve within the fresh run.
The selected and endpoint comparisons answer different questions and must not
be collapsed into one result. Also report every modality/task/source Accuracy,
NLL, Brier and ECE; minimum task accuracy; video question-type metrics; actual
consumption, unique rows/assets/groups; trainable/total parameters; wall,
training and evaluation time; peak VRAM/RAM; and selector warnings.

Retain the fresh-coverage policy for further research only if its selected
macro NLL is lower, selected macro Accuracy is higher, and no modality loses
more than 2 percentage points versus the repeat selected checkpoint. This is a
research rule, not a product threshold. Use the existing 5,000-resample,
seed-17 scene-cluster bootstrap for saved Video predictions and state its
development-selection limitations. Do not tune, add seeds, extend the budget,
or promote a model from this one comparison. The previously inspected
validation set is not a blind audit; all product-quality gates remain open.

## Run sequence and safety

1. Pass CPU tests for the shared 6,656-question probe construction and verify
   all configuration, corpus, cache, pool, plan, reference-run and checkpoint
   hashes. A CPU preflight records the complete 32,768-step order, LR series,
   unique-example/asset counts and source/task mixes before any GPU allocation.
2. Run a separate 32-update TRAIN-only smoke with the fresh pool. Confirm
   initialization, sampler, finite gradients, save/reload parity and projected
   wall time. Smoke metrics are not a quality result.
3. Only if the local GPU is free and all preflight/resource gates pass, run the
   single fresh 32,768-update fit from new initialization, then a separate
   exact best/final replay. Never resume the earlier 2,048-step run.

The Orbitune process is an independently running job. The user explicitly
authorized local model training with its cooperative yield mechanism. Its
command advertises `--yield-sentinel`; if using that path becomes necessary,
record the sentinel request, wait for the normal step-boundary exit and final
checkpoint, and verify that process is terminal and the GPU idle before starting
the smoke. Do not terminate it forcefully or share the GPU. If it cannot yield
cleanly, leave the model run unstarted and report the blocker.
