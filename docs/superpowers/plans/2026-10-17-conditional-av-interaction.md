# Conditional AV Interaction Implementation Plan

> Execute locally in this session under the user's existing autonomous-work
> authorization. Preserve the original baseline and all executed sources.

**Goal:** An opt-in, mechanically verified conditional AV residual, not a trained model.

**Architecture:** Wrap `NativeCandidateReadout`; add compact parallel bidirectional
exchange, candidate-conditioned independent gates and a zero-initialized residual.
Keep the original scoring and JSON contracts and explicit ablation policy.

**Tech Stack:** Existing Python3.11/PyTorch2.6; CPU only, no dependency changes.

**Spec:** `docs/superpowers/specs/2026-10-17-conditional-av-interaction-design.md`

## Global constraints

200M total before quantization; no old artifacts changed; no sealed audit; no real
training without complementary synchronized data admission. Add no runner default.

## Review focus

- Policy/checkpoint mismatch must fail instead of silently changing the model.
- Missing or masked evidence must not invent an audio/video partner.
- Zero residual initially hides gradients; test reach using a disclosed fixture.
- Candidate reordering must only reorder outputs and per-candidate diagnostics.
- Gate values are not quality scores; contract evidence is not scientific evidence.

## Tasks

1. Add `tests/test_native_av_interaction.py` before implementation. Assert initial
   and missing-input exact parity, invalid input rejection, conditional independent
   gates, permutation/mask contracts, bidirectional gradient reach, state round trip,
   policy rejection and encoder-inclusive budget. Run and retain missing-module RED.
2. Add `src/tiny_omni_decision/native_av_interaction.py` with
   `ConditionalAVReadout(baseline=None, *, exchange=True, gated=True)` and
   `forward(blocks, queries, *, diagnostics=False)`. Implement the exact spec,
   same-policy load and `parameter_ledger()` without touching the baseline file.
   Run the targeted tests GREEN, inspect all outputs, then commit.
3. Add an immutable CPU contract receipt outside Git: source/environment/config
   hashes, instantiated parameter ledger, targeted/full test outcomes and actual
   metadata-only AV observation count. No accuracy or trained gate interpretation.
4. Publish a compact report and a ROADMAP entry. Run Ruff/full CPU suite/manifest
   validation, review the diff, commit/push and check the exact commit's CI.

## Execution ledger

- 2026-10-17: inspected baseline, transport, previous gate failures and AV admission
  reports. Remote fetch matches local673932c. No Python run or GPU workload active.
- Ruling: prior explicit autonomous authorization covers design/implementation;
  no repeated permission request. Real joint learning remains data-gated.
- Ruling: keep the first stage local and mechanical; no new quality experiment is
  configured or scheduled on the unsuitable single-modality corpus.
- Tasks1/2: missing-module RED preserved; model/baseline/metadata tests GREEN36.
  Parallel exchange inputs, independent gate extremes and same-seed arm tensors
  checked. Implementation committed as e31cb35.
- Task3: CPU metadata audit PASS on e31cb35; total197,097,949 parameters. TRAIN
  17,341observations and validation3,843 have0AV observations. No learning admitted.
- Review ruling: `active_joint_parameters` counts policy-enabled allocation, not
  per-example nonzero gradients. Clarified in the report. Scalar shared correction
  bias cancels in option softmax; do not require a nonzero CE gradient for it.
- Task4 local verification: Ruff PASS,2,192CPU tests passed/1skipped,25manifest
  commands PASS;994prior artifact hashes unchanged. No original baseline or
  trainer file changed. Publication/CI receipt is outside Git to avoid a recursive
  commit-hash dependency. Quality learning remains unexecuted and unadmitted.
- CI1851 exposed an unnecessary optional-Torch import in metadata counting.
  Reproduced in a subprocess with Torch blocked, separated metadata schema from
  model construction, and added a runtime schema equality guard. RED/GREEN kept.
  Final full verification2,193passed/1skipped, Ruff and25manifest commands PASS.
