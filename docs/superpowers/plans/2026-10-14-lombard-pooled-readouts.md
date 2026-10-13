# Fixed Lombard Native Slot Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline, task-by-task. Steps use checkbox syntax.

**Goal:** Measure frozen native audio/video/joint command decisions with the pre-quality fixed protocol; preserve poor results and calibration/uncertainty evidence.

**Architecture:** A shared1154->64->51 MLP reads train-normalized384D speech/768D video means and2 presence bits. Split51 logits into6 complete native vocabularies. All three arms have identical initialization/sampling/budget and common source IDs, with encoders frozen outside training. Loading rejects partial feature jobs. Native JSON preserves all options without legacy-schema modification.

**Tech Stack:** Existing Torch2.6/NumPy/Python3.11; same local RTX3080. No installs or cloud compute.

**Spec:** configs/pretrained_reuse/lombard-grid-pooled-slot-v1.json (SHA047e98485298c5ff26900e20335aa49a1aa8cbf0581772482edc2dad3c19afe4); docs/pretrained_reuse/R2_LOMBARD_GRID_VISUAL_FEATURES_20261014.md.

## Global Constraints

- Preserve train4080/calibration398/validation889 source/speaker/phrase groups and SHA pins; all51 choices including25 letters.
- Seed17,24epochs,batch128,AdamW LR.001/weight_decay.001; no encoder training.
- Earliest minimum unrounded raw validation macro-NLL best, final separately; calibration-only17 geometric temperatures.25..4 per slot.
- Every attempted arm/smoke logged; unique versus repeated and total versus selected consumption distinct.
- No old artifact edits, sealed audit, artificial noise, extra seeds/epochs or metric-driven protocol changes.
- No final quality claim from smoke/pooling; no temporal/general Omni/product adoption claim.
- User explicitly authorized autonomous strategy/implementation; execute inline without another design approval.

## Review Focus

- A correctly sized partially allocated feature file must fail without PASS/IDs/hash/reload/immutability evidence.
- Reordered/duplicate/cross-split media identities or wrong provenance must fail before training.
- All native choices/macro denominators retained; JSON option permutation exact and ties canonical.
- Calibration cannot use validation; smoke cannot evaluate or select on held-out data.
- Best/final, sample accounting, checkpoint reload and paired controls must use the same frozen data and model.

### Task 1: Native readout, normalization, metrics and JSON

**Files:** create scripts/lombard_readout_core.py; tests/test_lombard_readout_core.py.
**Interfaces:** SlotHead.forward(B,1154)->(B,51); fit_normalizer(audio,video); make_inputs(audio,video,condition,normalizer); summarize(logits,targets,temperatures=None)->metrics,probabilities; json_decision(logits,option_orders=None,temperatures=None); selected_epoch(history); select_temperatures(cal_logits,cal_targets).

- [ ] Tests first: missing module RED; shape/finite/invalid targets, all51 options/count77235, train-only normalization, zero/mask semantics, perfect/uniform metrics and earliest raw-NLL selection, per-slot calibration and exact JSON option permutation/full25 letters.
- [ ] Implement fixed MLP and pure helpers. Expected: pytest task file PASS, Ruff PASS. Commit.

### Task 2: Frozen complete source/cache loader

**Files:** create scripts/lombard_readout_data.py; tests/test_lombard_readout_data.py.
**Interfaces:** load_data(root,smoke=False)->raw arrays by split,targets by split,rows,provenance; validate_bank(info,values,expected_rows,kind,media_rows)->None.
**Consumes:** Task1 vocab/target helpers; frozen source/protocol files and complete extraction outputs.

- [ ] RED tests for incomplete status/allocated cache, wrong shape/dtype/hash identity/order/duplicate/input SHA/source frames/parameter immutability; legitimate fixture passes.
- [ ] Implement source/license/corpus/media/archive/feature SHA gates. Full mode requires both entire banks; smoke uses only four predeclared train reference IDs and complete audio/native-smoke outputs. Expected: task tests/Ruff PASS. Commit.

### Task 3: Separate-process smoke and deterministic full trainer

**Files:** create scripts/train_lombard_grid_readouts.py; tests/test_lombard_readout_training.py.
**Interfaces:** fixed CLI --smoke only; full run writes new per-arm directories, best/final/history/predictions/normalizer/sampling hashes and counters. No budget/data overrides.
**Consumes:** Task1 core, Task2 loader.

- [ ] RED sampler/consumption/checkpoint-selection tests; implement identical seed17 orders and tensor initialization, finite gradient checks, fixed24 budget, validation-only selector, calibration-only temperature; best reload and native JSON permutation checks. Expected: task tests PASS.
- [ ] Run separate GPU smoke(three arms,two train-only updates,four reference assets), log VRAM/time/all retained artifacts. Expected: PASS execution only, no heldout metrics.
- [ ] Full run only after actual extraction process finishes and both caches pass gates; no re-launch while live. Expected: all3x24 epochs executed and best/final preserved, or concrete incomplete cause recorded without quality failure label.
- [ ] Commit tested implementation/smoke; incomplete full execution is explicitly not a completed experiment.

### Task 4: Fixed-model controls, uncertainty and result audit

**Files:** create scripts/report_lombard_grid_readouts.py; tests/test_lombard_readout_report.py; docs/pretrained_reuse/R2_LOMBARD_GRID_RESULTS_20261014.{md,json}.
**Consumes:** all three complete Task3 outputs, same validation IDs/order.

- [ ] RED tests for source-coherent different-speaker-group bijection, paired-bootstrap alignment/group preservation, saved prediction/metric/checkpoint/accounting consistency.
- [ ] Implement10,000 whole-group bootstrap/seed17/percentile95%; matched versus unimodal and frozen-weight mismatch/missing controls; raw/calibrated per-slot/macro/min/whole-utterance/p-l/speaker metrics. Expected: tests PASS, saved results reproduce all metrics.
- [ ] Run actual full reports, preserve negative outcomes and pooling/task/calibration limitations; minimal README/ROADMAP updates. Full tests/Ruff/manifest validation; commit/push/check target CI.
- [ ] Fresh final review under requesting-code-review on most capable available model; fix Important issues with RED->GREEN, no second review pass. Keep goal active until whole R0-R8 objective is proved.

## Execution

Pre-flight: Task1 dimensions/vocab/metrics feed Task2 arrays and Task3 logits without translation; Task2 prohibits incomplete video banks before Task3; Task3 saved IDs/logits/temperatures feed Task4 paired metrics. No shared-interface conflict.
