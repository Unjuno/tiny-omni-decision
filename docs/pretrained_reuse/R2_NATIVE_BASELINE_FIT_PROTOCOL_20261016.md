# First common native readout fit: fixed experiment

This is the first supervised fit of the196.255M native composition. It is not
Teacher/QAT continuation, product promotion, an xLSTM fit or a new audit. The
same frozen features and aligned candidates support all four modalities.

## Fixed policy before measuring quality

- Original native seed17 readout checkpoint; all889,345 readout parameters train.
  No encoder is loaded for updates. Frozen banks are immutable and hash-bound.
- Seed17, FP32, TF32 off, AdamW lr0.001, betas(0.9,0.999), eps1e-8,
  weight decay0.01, global gradient norm clip1.0. CE over **all supplied options**.
-2,048 optimizer updates,4 single-question microbatches per update, one each
  from text/image/audio/video. Average the four losses. No performance early stop.
  Warmup62updates; cosine decay from update62 to2,048. No mid-run budget changes.
- Within each modality rotate its cohorts equally; within a cohort rotate its
  tasks equally. Independently seed/shuffle each task bucket, exhaust it before
  reshuffling. Text cohorts are binding/numeric/legacy; audio Lombard/STARSS;
  image/video each one source. This is a declared baseline policy, not a proven
  optimum. Natural event positive/negative frequencies remain unchanged.
- Preserve every row in the corpus. Log the full8,192-row consumption order,
  repeated questions, unique observations, groups and task/source counts before
  running. The512 video questions necessarily repeat four times; disclose this.
  No source/option simplification or invalid unique-coverage claim.
- Deterministic per-microbatch option permutation, with target and IDs reordered
  together. Initial checkpoint, order and permutation policy shared by future
  architecture comparisons.
- Evaluate the **complete24,346-question development validation** at steps
  0,256,512,768,1024,1280,1536,1792,2048. Also evaluate a fixed TRAIN probe of up to
  16questions per task (source+task), chosen by seeded hash before training.
  This probe is descriptive, not a matched-distribution generalization estimate.
- Save per-example logits with original IDs/option order and target indices.
  Report overall/per-modality/per-source/per-task Accuracy/NLL/Brier/ECE;
  STARSS additionally reports positive counts, recall/specificity and balanced
  accuracy when both classes exist. Zero-positive classes remain explicitly
  unmeasured for positive recall. Global sample-weighted Accuracy is insufficient.
- Research checkpoint selector: lowest equal-task-within-source,
  equal-source-within-modality, equal-four-modality **macro NLL**, min_delta1e-5,
  earliest checkpoint on a tie. Retain best and final separately. This research
  selector does not relax product/retention gates or promote the result.
- Log train rolling CE/Accuracy, fixed TRAIN probe metrics, validation metrics,
  min modality/task Accuracy, LR, gradient norms, timings and VRAM. Warn on falling
  train probe NLL with rising validation NLL or improving aggregate with degrading
  weakest modality. Warnings do not terminate the planned learning curve.
- Save full optimizer state, step, selector, CPU/CUDA/Python RNG states and
  sampler order at each scheduled checkpoint. No resume is attempted in this
  unit, and exact interruption/resume equivalence remains unverified.

## Admission and safety

First perform a separate32-update TRAIN-only smoke with this same optimizer
and schedule, then discard its learned weights. Measure sec/update, memory,
finite gradients and exact save/reload; use original initialization and seed
again for the actual run. The smoke must not inspect validation performance.
Proceed only if the full run is projected to fit the2hour local budget.
No other GPU process may be active at admission. Stop on NaN/Inf, input/source
hash change, 6GiB allocated CUDA,10GiB process peak RAM or2GiB disk reserve.
Interrupted/failed execution is incomplete, not evidence against model capacity.

After completion recompute best and final predictions in a fresh process under
the identical pinned sources and require exact agreement. Compare fixed endpoint
and selected checkpoint without mixing claims. The old47-slice retention results
and failed prior candidates remain intact. All-task95–98%, genuine joint/OOD,
calibration and deployment gates remain open. LSTM and a nearly parameter-matched
nonrecurrent control use the same corpus/policy if the comparison proceeds;
new architecture performance is not inferred from this baseline fit.
