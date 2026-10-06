# Decision Re-specialization Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synchronize the repository's roadmap, recovery configuration, README, and config tests with the approved LM-to-Decision design: ternary bottleneck, two-stage Decision Recovery → Decision Re-specialization, and a 95% final gate for every modality.

**Architecture:** Keep the current high-precision Master and ternary conversion boundaries, but redefine post-quantization training as two explicit stages. The configuration records an initial candidate schedule and the required A/B/C/D/E causal comparison; no GPU trainer implementation is added in this sync.

**Tech Stack:** Markdown, YAML, Python 3.11, pytest, existing `load_structured_file`.

**Spec:** `docs/superpowers/specs/2026-10-06-compression-first-video-specialization-design.md`

## Global Constraints

- Final product gate: Text >=95%, Image >=95%, Audio >=95%, end-to-end Video >=95%.
- The 95% gate does not block ternary conversion.
- The frozen ternary base remains unchanged during the first Recovery/Re-specialization experiment.
- Full-vocabulary KL stays disabled by default.
- Recovery Teacher option distributions are an anchor, not a performance ceiling.
- Dense Recovery state must not be merged into ternary weights for deployment without explicit re-quantization and re-evaluation.
- Historical experiment metrics and sealed-audit records are not rewritten.
- This change does not implement ternary kernels, GPU Recovery training, or the temporal Video aggregator.

## Review Focus

- Recovery config must unambiguously encode two stages rather than one fixed loss.
- The fixed-recovery baseline must remain representable for comparison with annealed Re-specialization.
- The config must preserve option-only teacher signals and disabled full-vocabulary KL.
- README and ROADMAP must agree on Variant A/B/C/D meanings and all-modal 95% final gate.
- Tests must fail if the stage names, loss endpoints, or A/B/C/D/E comparison contract drift.

---

### Task 1: Synchronize ROADMAP and README terminology and phase flow

**Files:**
- Modify: `ROADMAP.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: approved design spec terminology.
- Produces: one repository-level narrative used by workers for later implementation.

- [ ] **Step 1: Update the ROADMAP top-level flow**

Replace the single `Recovery` step with:

```text
Decision Recovery
        ↓
Decision Re-specialization
```

Define Variant A as the ternary Decision Core after both stages. Keep Variant B as lightweight Decision readout, Variant C as temporal Video + final all-modal Decision specialization, and Variant D as optional pooling/resampling.

- [ ] **Step 2: Rewrite Phase 8 around two-stage post-quantization training**

Phase 8 must state:
- Stage 1 Recovery uses stronger teacher anchoring plus GT CE/Brier.
- Stage 2 Re-specialization reduces KD pressure and increases GT pressure.
- exact weights are development-selected;
- fixed teacher-heavy Recovery and annealed Recovery→Re-specialization are both required candidates;
- Master is an anchor, not a ceiling.

- [ ] **Step 3: Add the required A/B/C/D/E causal comparison to ROADMAP**

Use these exact labels:
- A — Master
- B — Ternary raw
- C — Ternary + fixed Recovery
- D — Ternary + Recovery→Re-specialization
- E — High-precision control

Document the interpretations `C > B`, `D > C`, `D > A`, and `D > E`.

- [ ] **Step 4: Update README's planned compression flow and variant definitions**

README must match the new roadmap and remove the stale definition where Variant C is pooling/resampler.

- [ ] **Step 5: Verify documentation consistency**

Search both files for stale phrases that imply:
- one-stage Recovery only;
- Variant C = pooling;
- final gate = 90%;
- Video-only 95% stretch.

Expected: none remain outside historical result descriptions.

- [ ] **Step 6: Commit**

```bash
git add ROADMAP.md README.md
git commit -m "docs: align roadmap with decision re-specialization"
```

### Task 2: Replace fixed Recovery loss config with explicit two-stage policy

**Files:**
- Modify: `configs/recovery/probability_distillation.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `load_structured_file(path) -> dict[str, Any]`.
- Produces: a declarative Recovery/Re-specialization policy for later trainer implementation.

- [ ] **Step 1: Write the failing config test**

Add `test_recovery_config_declares_two_stage_decision_respecialization_policy()` asserting:
- `schema_version == 1`
- `recovery.method == "lora"`
- `teacher_signal.cache_scope == "single_decision_position_options_only"`
- `teacher_signal.store_full_vocabulary_logits is False`
- stage order is exactly `["recovery", "respecialization"]`
- Recovery stage weights are `option_kl=1.0`, `cross_entropy=0.2`, `brier=0.2`
- Re-specialization endpoint is `option_kl=0.2`, `cross_entropy=1.0`, `brier=0.2`
- comparison candidates are exactly `["master", "ternary_raw", "fixed_recovery", "recovery_to_respecialization", "high_precision_control"]`
- `full_vocab_kl == 0.0`

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```bash
pytest tests/test_config.py::test_recovery_config_declares_two_stage_decision_respecialization_policy -q
```

Expected: FAIL because the current config has a single fixed `loss` block and no stage/comparison policy.

- [ ] **Step 3: Update `probability_distillation.yaml`**

Keep the existing base, LoRA, and teacher-signal settings. Replace the one-stage loss policy with:

```yaml
post_quantization_training:
  stage_order: [recovery, respecialization]
  recovery:
    option_kl: 1.0
    cross_entropy: 0.2
    brier: 0.2
  respecialization:
    schedule: linear
    option_kl_start: 1.0
    option_kl_end: 0.2
    cross_entropy_start: 0.2
    cross_entropy_end: 1.0
    brier: 0.2

loss:
  full_vocab_kl: 0.0

experiment_comparison:
  candidates:
    - master
    - ternary_raw
    - fixed_recovery
    - recovery_to_respecialization
    - high_precision_control
```

Add notes that these are initial candidate values, selected/validated on clean development data before becoming a training default.

- [ ] **Step 4: Run the targeted config test**

Run:

```bash
pytest tests/test_config.py::test_recovery_config_declares_two_stage_decision_respecialization_policy -q
```

Expected: PASS.

- [ ] **Step 5: Run all config tests**

Run:

```bash
pytest tests/test_config.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/recovery/probability_distillation.yaml tests/test_config.py
git commit -m "config: stage recovery and decision respecialization"
```

### Task 3: Repository-wide consistency verification

**Files:**
- Verify: `ROADMAP.md`
- Verify: `README.md`
- Verify: `configs/recovery/probability_distillation.yaml`
- Verify: `docs/superpowers/specs/2026-10-06-compression-first-video-specialization-design.md`

**Interfaces:**
- Consumes: outputs from Tasks 1 and 2.
- Produces: a repository state where docs/config communicate the same pipeline.

- [ ] **Step 1: Run CPU tests**

Run:

```bash
pytest -q
```

Expected: all CPU tests pass.

- [ ] **Step 2: Run lint**

Run:

```bash
ruff check .
```

Expected: PASS.

- [ ] **Step 3: Verify exact final quality contract**

Search tracked docs/config for active product-target language.

Expected active contract:
- Text >=95%
- Image >=95%
- Audio >=95%
- end-to-end Video >=95%

Historical 90% experiment records may remain in historical result documents but must not appear as the current product gate.

- [ ] **Step 4: Verify no implementation claim is overstated**

README/ROADMAP must still state that ternary runtime, Recovery/Re-specialization GPU training, and temporal Video specialization are planned/unimplemented until actual experiments exist.

- [ ] **Step 5: Commit any final consistency-only corrections**

```bash
git add ROADMAP.md README.md configs/recovery/probability_distillation.yaml tests/test_config.py
git commit -m "docs: finalize lm-to-decision pipeline contract"
```
