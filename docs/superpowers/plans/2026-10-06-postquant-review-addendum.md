# Post-quantization plan review addendum

**Scope:** Planning corrections reconciled with remote commit `b049c28580a3c62ea6bd770b480ff5fc81b60f19`. Read with the [execution plan](2026-10-06-post-quantization-respecialization-experiment.md), [sync plan](2026-10-06-decision-respecialization-sync.md) and [design spec](../specs/2026-10-06-compression-first-video-specialization-design.md). This does not implement code or authorize training.

The concurrent remote work is preserved. Do not replace its 1,024-update/256-update stage split, 8-update standalone smoke, group-size-128/threshold-0.7 quantizer candidate, `PostQuantConfig` interfaces or Q0-Q4 naming with an unreviewed alternative. These are experimental defaults, not literature-proven optima. Final Text/Image/Audio/end-to-end Video Accuracy remains >=95%; compression-entry policy B remains unchanged.

## Corrections workers must apply

### 1. Dependency order is operational, not just document order

Implement CPU policy, loss, cache and quantizer tests first. Task 5's one-layer quantizer smoke is independent; its 8-update end-to-end smoke depends on the real runner in Task 6 and runs only after Task 6 is implemented and tested. Do not report that smoke as executed merely because Task 5 appears earlier. The smoke uses separate hashes/artifacts and is not resumable as the 1,024-update comparison.

### 2. Matching and per-arm identity are different

Keep a per-arm effective-config/run hash for exact resume. Also compare a canonical shared-training-contract hash that excludes only fields intentionally different across arms, such as arm ID, output path and documented base precision. Q2's fixed coefficients are an intentional difference from Q3; Q3/Q4 coefficients must match. Do not demand identical complete manifests for different arms, and do not exclude optimizer, initialization, data/order, temperature or target topology from the shared comparison.

Store the initial adapter tensors once and verify their hashes after loading each arm. Reset RNG after setup/evaluation. Sample ordering and dropout RNG depend on setup too; identical seed labels alone are insufficient. Same-update comparisons are primary; independently chosen best checkpoints are secondary.

### 3. Release and scientific explanation remain separate

The existing research adoption thresholds, at least 2 percentage points minimum-modality gain over fixed Recovery and at most 1 point regression in another modality, remain initial predeclared research rules. They are not the final 95% gate and do not prove generalization from one noisy development set. Keep UNCERTAIN results and all failures. A ternary artifact can be useful without exceeding Q4 if it meets final quality and real deployment budgets.

Beating Q4 is evidence for the combined ternary intervention under matched training, not selective deletion of generation circuitry or an effect attributable to zeros alone. A failed rank-16 frozen-base attempt does not disprove all low-bit training. Do not automatically add QAT, bigger adapters or external teachers without a separate versioned experiment and resource check.

### 4. Freeze input and evaluation contracts without manufacturing answerability

The limited single-media guard must reject every unsupported extra input before processor invocation. True simultaneous-media support is mandatory before fusion training/claims, not optional afterthought validation. Pin the declared combinations and verify every reference reaches the processor/model.

Temporal windows carry timestamps and source identity. Video-level labels supervise the aggregate output. Audio hard negatives must remain genuinely wrong and unambiguous under a prespecified construction/review rule. Counterfactual swaps and modality removal do not automatically preserve labels. A missing-evidence case may need an insufficient-evidence decision; do not force it to count as an ordinary correctly labeled input. Masking/deletion causing an accuracy drop by itself is not proof of fusion because it can also change the input distribution.

### 5. Do not call a reference model a packed deployment

Reference codes/scales, expanded BF16 weights, residual LoRA and high-precision encoders have separate byte and memory costs. Record the actual loaded representation and decoder/attention backend. Preserve one current backbone adapter state; small temporal/readout modules count too. Dense residual merging requires explicit re-quantization and new quality checks.

Before release freeze target hardware and numeric latency, memory, size and startup limits. For video separate observation-window delay from processing delay and include decode, all clips and aggregation. Missing runtime budgets mean deployment qualification is incomplete; they do not prevent bounded quality research.

## Implementation evidence required next

- A strict schema-v2 loader rejects unknown/legacy fields; old fixed-loss config remains unchanged.
- Actual trainer tests show coefficient changes reach the numerical loss, not only YAML parsing.
- Loss tests cover 2/4/10/60/62 choices, padding, permutations, teacher detachment and accumulation.
- Stage-boundary resume tests reproduce coefficients, sample order and optimizer/RNG state.
- A separately authorized 8-update GPU smoke passes before the three trained 1,024-update arms run.
- Reports distinguish documentation synchronized, CPU tests passed, GPU smoke verified, bounded experiment evaluated and final product qualified.

The 95% final audit remains untouched during all of this work. Report point estimates, denominators and asset-aware intervals; do not imply a population confidence guarantee from the point estimate alone.
