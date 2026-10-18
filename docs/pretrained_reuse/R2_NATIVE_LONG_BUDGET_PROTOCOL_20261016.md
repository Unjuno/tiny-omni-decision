# Native fixed-pool long learning curve

## Reason and priority change

The user's 2026-10-16 correction is accepted: short runs can miss delayed
generalization. Both recent coverage arms selected their final2,048 checkpoints
and still improved NLL late. Their low TRAIN accuracy also leaves undertraining
open. The recorded coverage retention failure remains valid at that budget; it
does not establish permanent forgetting, a capacity ceiling or a clean plateau.
This prospective longer curve takes priority over the proposed gradient audit
and architectural branching. Prior artifacts and historical reports remain intact.

[Power et al.](https://arxiv.org/abs/2201.02177) demonstrate delayed generalization
after memorization on small algorithmic data. [Omnigrok](https://arxiv.org/abs/2210.01117)
also studies image/language tasks and weight-norm/regularization effects. These
are reasons to measure a longer curve, not evidence that our model will grok or
reach95%. Current TRAIN performance has not reached the memorization stage.

## Frozen before execution

- One seed17 run, original native initial weights and fresh AdamW. No trained
  checkpoint is used as initialization; the separate32-step smoke is discarded.
- Existing attention readout889,345 trainable /196,254,874 total prequantization
  parameters; native feature encoders remain frozen. Same FP32, deterministic
  kernels, TF32 off, complete candidates, CE/8 and two examples/modality/update.
- Use the previous `repeat` pool/independent occurrence plan, repeated for16
  exact cycles. No added rows, labels, task weights, features, frames or modules.
  Cycle zero preserves the original sample and candidate-shuffle prefix. Later
  cycles use globally unique occurrence numbers for the same shuffle rule.
-32,768 updates /262,144 presentations,6,656 unique questions,255,488 repeats;
 65,536 presentations/modality. Unique Text/Image/Audio2,048 each; Video512.
  Video appears128 times/question; other modalities32. Unique observations5,107,
  media3,114 and groups3,382 are unchanged. Repetition cannot repair missing labels.
- AdamW lr.001, betas(.9,.999), eps1e-8, wd.01, global clip1. Warmup62 then cosine
  over32,768 from the start, final LR0. Unlike the old schedule, LR remains
  positive after2,048. Save the entire LR vector before training. Changing cosine
  horizon changes the earlier LR trajectory: this is NOT a steps-only comparison
  against the old2,048 fit, and is not a resume.
- No performance early stopping or adaptive schedule/seed/architecture changes.
  Safety stops only: nonfinite loss/gradient/output, integrity failure, GPU/RAM/
  disk limit, or two-hour resource deadline. An interrupted run is incomplete.

## Measurements and interpretation

Evaluate steps0,512,1024,2048,4096,8192,12288,16384,24576,32768. At EVERY point:

- All6,656 unique questions in the fixed TRAIN pool, original candidates. This
  is fixed inference on TRAIN, unlike rolling microbatch metrics or the former
  mostly unconsumed1,456-question probe. Report cumulative exposure separately.
- All24,346 existing validation questions, original order/options/preprocessing.
  Keep every one of71 tasks; do not trim28 validation-only tasks. Report43 common
  task gaps separately, with48 TRAIN-only tasks and prevalence differences noted.
- Overall, hierarchical macro, modality/source/task Accuracy/NLL/Brier/ECE,
  minimum task/modality accuracy, temporal-video subset, STARSS positive recall
  and specificity, per-example logits with IDs/targets and parameter L2 norms.
- Per-update CE/Accuracy (explicitly rolling), LR, unclipped gradient norm,
  duration, actual sample/occurrence IDs; cumulative unique/repeat/asset counts.
- Select best by unchanged hierarchical macro NLL/min_delta1e-5/earliest tie;
  step0 eligible. Preserve every scheduled checkpoint, best and final separately.
  Persist optimizer/RNG, but runner has no resume API; do not claim strict resume.

Primary contrast is THIS run's fixed2,048 versus32,768. Auxiliary8,192/16,384/
24,576 show whether improvement persists. Show every measured trade-off; improved
macro alone does not authorize a modality sacrifice. Report the old repeat fit
only as historical reference with a different LR trajectory.

Separate ordinary continued learning from delayed generalization: if TRAIN is
still poorly fitted, say underfitting persists. If TRAIN becomes well fitted
before validation improves, report the measured delay/task, without declaring
a universal grokking mechanism or causal proof. No improvement within32,768
does not prove an architecture ceiling or rule out all later learning.

Diagnose Video2,048→32,768 from paired scene predictions:5,000 bootstrap draws,
seed17, percentile95% linear quantiles; also temporal-only. All are observed
development metrics with checkpoint-selection effects, not blind evidence.
At the endpoint retain the existing1,250 TRAIN state-pair diagnostic and12,800
TRAIN-union diagnostic; only6,656 parent rows receive optimization here.

Fresh-process replay must exactly reproduce all best/final validation and fixed
TRAIN arrays plus endpoint state/union arrays. Keep sealed audit/calibration
unloaded; no product promotion, new encoder training, LSTM/xLSTM or quantization.
The200M and every-required-task95–98%, temporal/joint/OOD goals remain unmet.

## Commands and local resources

New config:`configs/pretrained_reuse/native_long_budget_v1.json`.
New runner:`scripts/train_native_long_budget.py`; each output directory must not
exist. Pin clean committed source and all inputs before/after; no edits while live.

```powershell
python scripts/train_native_long_budget.py --phase smoke --policy repeat --output <new-smoke>
python scripts/train_native_long_budget.py --phase train --policy repeat --output <new-fit> --reference <smoke> --reference-sha256 <smoke-receipt-sha>
python scripts/train_native_long_budget.py --phase replay --policy repeat --output <new-replay> --reference <fit> --reference-sha256 <fit-receipt-sha>
```

Use local RTX3080 Laptop only. No GPU competition;6GiB allocation/10GiB process
working-set/2GiB free-disk/two-hour process guards. Previous2,048-fit training
alone took114s; evaluation dominated. Project32,768 training near30min, plus
full evaluation and replay; measure rather than promise this timing. The smoke
must confirm runtime below the bound. Cloud cost0. No automatic further extension.
