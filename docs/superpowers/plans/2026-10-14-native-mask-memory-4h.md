# Native mask-memory resource amendment implementation plan

> **For agentic workers:** Use superpowers:executing-plans; implement directly.

**Goal:** Execute the full frozen comparison under a separately recorded local4h budget.

**Architecture:** Preserve original native runner and all old artifacts. New entrypoint
rechecks old numerical admission and directly reuses unchanged native functions;
new CPU auditor/scorer checks products after inference, without annotation inputs.

**Tech Stack:** Existing Python3.11.9/Torch2.6.0+cu124/Transformers5.6.2/NumPy/PyAV.

**Spec:** `docs/pretrained_reuse/R2_NATIVE_MASK_MEMORY_4H_PROTOCOL_20261014.md`.

## Global constraints

14400s main/600s fresh, unchanged6GiB GPU/4GiB process/8GiB disk reserve. All696
seeds/8videos/twoarms/32times, identical numerical conditions. No legacy edits,
cloud/training/GT seeds/official evaluation/sealed audit/main merge or promotion.

## Review focus

1. Changing old media/model/library/reader bytes must reject admission.
2. Missing/reordered raw smoke products cannot admit execution.
3. Original2h failure cannot be renamed a success.
4. Incomplete main rows cannot admit fresh numerical replay.
5. Timeout/numerical/prefix faults preserve incomplete products and never restart.

### Task1: New entrypoint and strict reference validation

Create `scripts/probe_native_mask_memory_4h.py`, test
`tests/test_native_mask_memory_4h.py`. Interface
`validate_amendment(prior:dict,current:dict)->dict` allows only original source
commit to differ, rejects changed conditions/pins, recomputes projection and
returns explicit original/new resource gates. Concrete tests: unchanged contract
with differing commit accepted; changed frame count/pin rejected; nonfinite or
over14400 projection rejected; original gate remains false. Write tests, run RED,
implement, run GREEN. Reuse old native decode/execute/save/evidence/admit_main,
revalidate raw192 archive products via existing independent pure NumPy function.
Review, full CPU/Ruff/manifest checks, commit before actual GPU execution.

### Task2: Actual main and fresh prefix

Launch new main output directory, confirm PID/creation/command/GPU progress.
Never commit/edit inherited pinned implementation while active. Preserve all
16 final artifacts and freeze/RNG/hashes; after terminal success launch fresh
separate process into new output, first2seeds/botharms, exact raw/packed comparison.
Incomplete experiments remain incomplete, with saved reason and progress.

### Task3: Independent audit, proxy comparison and publication

Create CPU-only complete mask-product auditor + fixed-IoU group scorer/tests.
Verify all44544 products/16artifacts/cohort/time/pins; fresh scope2/696 stated.
Only after inference admission open unchanged8 TRAIN proxy rows. Keep all counts,
unknowns/ambiguous/unavailable and source group coverage, no semantic quality claim.
Record actual results/resources/unperformed gates. Minimal README/ROADMAP note,
commit/push small metadata/code/tests; check exact-head Actions and leave goal active.
