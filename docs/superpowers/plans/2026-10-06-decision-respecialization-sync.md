# Decision Re-specialization Sync — Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` task by task. Unchecked tasks are future implementation. This revision adjusts the plan; it does not declare the code, config migration or GPU training complete.

**Goal:** Align the current roadmap and README with the approved compression-first direction and link them to a real, testable post-quantization execution contract.

**Architecture:** Keep documentation alignment separate from executable training. Implement the new versioned policy, losses and runner through the companion experiment plan; never substitute a YAML-key test for a training integration test.

**Tech Stack:** Markdown and the existing Python/pytest project. No new runtime dependencies for this synchronization.

**Spec:** [LM-to-Decision design](../specs/2026-10-06-compression-first-video-specialization-design.md).

**Execution contract:** [Post-quantization experiment plan](2026-10-06-post-quantization-respecialization-experiment.md).

## Global constraints

- Final Text, Image, Audio and end-to-end Video Accuracy must each reach at least 95%; this is not a compression-entry gate.
- Keep the current high-precision run, frozen corpus, historical metrics, model weights and observed audit records unchanged.
- Maintain one shared Omni product. Single-media compression experiments do not satisfy simultaneous-media fusion requirements.
- Compression-induced regularization and selective loss of generative freedom are hypotheses, not guaranteed explanations.
- The first candidate is a frozen ternary reference core with an accounted-for adapter; dense correction does not make the entire effective model sparse.
- A genuine packed runtime, Video aggregation and full multi-media processing are separate downstream deliverables. No speed or product-completion claim from documentation changes.
- Preserve the valid refinements already present at `73f30b64f4e2e78f7994f3a0b764fa1a1bb036e8`; reconcile new remote changes before writing and never force-push.

## Review focus

1. Do not confuse causal arms Q0–Q4 with product Variants A–D.
2. Do not overwrite the legacy v1 Recovery config with unsupported schedule keys.
3. Do not announce a two-stage learner before the real runner, losses and resume tests exist.
4. Do not make successful sparsity regularization or superiority to the high-precision control a mandatory product criterion.
5. Do not let a fallback single-media input path silently drop evidence or become a claim of Omni fusion.

## Task 1 — Align ROADMAP and README

**Files:** Modify `ROADMAP.md`, `README.md`. Do not rewrite historical experiment reports.

**Interface:** Repository navigation and stage definitions; the companion experiment plan is the detailed implementation contract.

- [ ] Snapshot the target branch HEAD, read both documents and the two linked planning documents, and verify the companion plan actually exists. Preserve concurrent changes.
- [ ] Update the active flow to Master freeze, LoRA merge, ternary feasibility/damage measurement, fixed-Recovery versus Recovery-to-Re-specialization comparison, selected Variant A, equivalent option readout for Variant B, temporal Video plus genuine joint specialization for Variant C, and optional pooling/re-specialization for Variant D.
- [ ] Describe the two-stage policy as a candidate. If fixed Recovery wins, retain it as the reference; do not promote a losing annealed run merely to satisfy a diagram. The final product still requires all four modalities at least 95%.
- [ ] Replace the active fixed 2–20-option limitation with the current 2–62-option contract, including the 60-intent Text task. Preserve ordered option semantics and separate readout changes from transformer attention changes.
- [ ] Add the input contract gate: unsupported supplied media must fail closed; simultaneous-media learning and ablation/contradiction evaluation are mandatory before claiming fusion. This does not block a clearly labeled single-media compression pilot.
- [ ] Link to the companion plan and mark ternary reference execution, packed execution, policy/loss/runner implementation and GPU experiments with their actual status. Documentation sync is not their completion.
- [ ] Inspect the diff for stale Variant C-as-pooling, Video-only 95%-stretch, or current 90% shipping language. Historical 90% targets remain historical; do not falsify them.
- [ ] Run the existing documentation checks if present and `git diff --check`; commit only the intended documentation changes.

## Task 2 — Use unambiguous comparison names

**Files:** Modify active comparison descriptions in `ROADMAP.md` and `README.md`; verify the linked spec and execution plan.

**Interface:** Exact arm IDs used by the later runner and result reports.

| Experiment ID | Role |
|---|---|
| `q0_master` | Frozen merged high-precision Master, evaluation only |
| `q1_ternary_raw` | Quantized Q0, evaluation only |
| `q2_fixed_recovery` | Q1 with fixed Recovery objective |
| `q3_recovery_to_respecialization` | Q1 with staged objective |
| `q4_high_precision_control` | Q0 with the same staged objective and matched adaptation |

- [ ] Remove the old A/B/C/D/E causal-arm aliases from active instructions. Do not rename historical candidate artifacts or product Variants A–D.
- [ ] State that only Q2/Q3/Q4 are training runs. Match initial adapter tensors, data/order, gradients per update, optimizer/LR, coefficients where appropriate, validation and selection policy; matching only update count is insufficient.
- [ ] Describe Q3 versus Q2 as the schedule comparison and Q3 versus Q4 as the quantization-intervention comparison. Neither isolates zero rate alone. Q3 can be a useful compact product without beating Q4 if it meets final quality and deployment gates.
- [ ] Verify all active names and links agree; run `git diff --check` and commit the naming changes with Task 1 when practical.

## Task 3 — Safe config handoff, not a config-only false completion

**Files in the execution plan:** Create `src/tiny_omni_decision/postquant_policy.py`, `tests/test_postquant_policy.py`, `configs/recovery/decision_respecialization_v2.example.yaml`, then the loss/cache/reference-quantizer/runner/comparison files specified there. Unqualified production module names in that plan are inside `src/tiny_omni_decision/`, not the repository root.

**Interface:** The strict schema-v2 loader and per-update `LossWeights` consumed by the actual runner.

- [ ] Keep `configs/recovery/probability_distillation.yaml` unchanged as the legacy template during documentation synchronization. The former proposal to retain schema v1 while replacing its loss keys is superseded.
- [ ] Implement the new schema and its example config together with a strict parser under Task 2 of the execution plan. Reject legacy schemas and unsupported fields rather than silently falling back to fixed weights. Leave historical Decision-training configs and `tests/test_config.py` expectations intact.
- [ ] Implement and test the categorical loss reductions, real coefficient consumption and stage-boundary resume under Tasks 3 and 6. No claim of two-stage training from successful YAML loading alone.
- [ ] Use the companion plan's explicit initial budgets and boundaries; they are experiment defaults, not proven optima or a launch authorization. A smoke run has its own config and artifacts and must not be resumed as a larger run.
- [ ] Complete CPU tests before GPU work. The one-layer quantizer checks in Task 5 may precede the runner; its 8-update end-to-end smoke must wait until Task 6 is implemented. No full experiment before all prerequisite checks and resource caps pass.
- [ ] Record separate statuses: documentation synchronized, CPU implementation tested, GPU smoke verified, bounded experiment evaluated, and final product qualified. Advance only the status supported by executed evidence.

## Acceptance for this synchronization

Active docs agree on the final four-modality 95% gate, compression-entry policy, Q0–Q4 names, Variant A–D roles, no-silent-media-drop rule, and the distinction between numerical reference and packed deployment. All referenced plan files exist. Legacy configs, active training and historical results remain intact.

Completion of this synchronization does not complete the companion experiment or the product. Report actual changed paths and commit IDs, tests really run, and any remaining implementation gates.
