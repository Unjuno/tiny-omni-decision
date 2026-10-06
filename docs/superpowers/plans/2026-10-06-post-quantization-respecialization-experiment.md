# Post-quantization Decision Re-specialization Experiment — Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to implement this plan task by task. Unchecked items are future work, not completed implementation or permission to start GPU training.

**Goal:** Test fixed Recovery against Recovery-to-Re-specialization and an otherwise matched high-precision control, without treating compression as proof of a quality gain.

**Architecture:** Implement a separate, fail-closed post-quantization experiment path. Reuse the frozen Decision Master, option interface, data integrity utilities and evaluator; do not change an in-flight high-precision run. First establish a reference ternary implementation for quality research, not a claimed fast deployment backend.

**Tech Stack:** Existing Python, PyTorch, PEFT, Pydantic/YAML and pytest environment. Record installed versions and attention/decoder backend; do not upgrade dependencies silently.

**Spec:** [LM-to-Decision design](../specs/2026-10-06-compression-first-video-specialization-design.md).

**Planning baseline:** repository commit `73f30b64f4e2e78f7994f3a0b764fa1a1bb036e8`. The accompanying [sync plan](2026-10-06-decision-respecialization-sync.md) is documentation alignment, not a substitute for this executable implementation.

## Global constraints

- Final product: Text, Image, Audio and end-to-end Video must each achieve at least 95% Accuracy on the preregistered final evaluation. Do not replace this with macro accuracy, teacher agreement or selective coverage.
- The final 95% threshold does not block a bounded compression experiment. Engineering completion and product completion are separate statuses.
- Preserve historical results, base/model revisions, active-run config, corpora, predictions and checkpoints. No direct main writes, force pushes or automatic training restarts.
- Use Q0–Q4 for experiment arms; reserve Variant A–D for product artifacts. Historical Video Teacher candidates retain their original names.
- Default teacher signal is option logits, not full-vocabulary output. Option KL already targets Decision behavior; it does not require reproducing long-form generation.
- Selective removal of unnecessary generative capability and improved generalization are hypotheses, not measured mechanisms or guaranteed effects.
- Keep the first ternary base frozen. Count dense adapter overhead separately; do not call the effective model sparse or merge dense updates into ternary weights without re-quantization and re-evaluation.
- No final/sealed-audit access for training, quantizer calibration, coefficient tuning or checkpoint selection.
- No paid compute or full experiment launch until its artifact paths, data rights, environment and resource caps are frozen in a run manifest. A planning document is not such a manifest.

## Review focus

1. Supplied media that the processor cannot use must be rejected, not silently discarded (Task 1).
2. A new YAML schedule must be read by the actual trainer; old schemas, unknown fields and missing stage budgets must fail (Tasks 2 and 6).
3. Different option counts, option permutations, padded choices and stale cache entries must not change loss semantics or teacher-target alignment (Tasks 3 and 4).
4. A stage-boundary resume must preserve coefficients, example order, optimizer/LR state and next update, without resetting adaptation (Task 6).
5. Dequantized reference execution, changed initialization, different exposure or omitted failures must not masquerade as low-bit speed or a causal quantization gain (Tasks 5 and 7).

## Frozen comparison design

| Arm | Base and additional training | Purpose |
|---|---|---|
| Q0 / `q0_master` | Merged, frozen high-precision Master; evaluation only | Master reference and teacher cache |
| Q1 / `q1_ternary_raw` | Exact Q0 quantized by the recorded recipe; evaluation only | Raw quantization damage |
| Q2 / `q2_fixed_recovery` | Q1 plus a new adapter; fixed KL/CE/Brier weights | Fixed-Recovery control |
| Q3 / `q3_recovery_to_respecialization` | Q1 plus the identical initial adapter; staged weights | Re-specialization candidate |
| Q4 / `q4_high_precision_control` | Q0 plus the identical initial adapter; the Q3 schedule | Non-quantized additional-training control |

Only Q2, Q3 and Q4 require training. They use identical adapter tensors at initialization, resolved target paths/rank, optimizer, LR schedule, sample IDs/order, label permutations, gradient accumulation, evaluation IDs, checkpoint-selection rule and update budget. Match tensors by hash, not only a nominal seed. Use the same frozen Q0 teacher cache for all three.

Compare Q3 with Q2 for the coefficient-schedule effect, and Q3 with Q4 for the effect of the quantization intervention under this recipe. Neither comparison alone proves that zeros, rather than altered values or another quantization property, caused the result. Beating Q4 is not required for a useful compact product.

### Initial candidate parameters — not empirically optimal values

| Field | Meaning / 意味 | SI unit | Definition and range | Type |
|---|---|---|---|---|
| `total_updates` | 追加optimizer更新数 | 1 (count) | 1,024 in the first bounded candidate; positive integer | Integer scalar |
| `recovery_updates` | 第一段階の更新数 | 1 (count) | 256; strictly less than `total_updates` | Integer scalar |
| `completed_updates` | 次の更新前の完了数 | 1 (count) | Zero-based state; 0 through 1,023 for training | Integer scalar |
| `temperature` | 蒸留温度 | 1 | 1.0 for this experiment; reject other values until separately specified | Real scalar |
| `option_kl` | 選択肢KLの係数 | 1 | Q2: 1.0 throughout; Q3/Q4: 1.0, then linear decay to 0.2 | Nonnegative real scalar |
| `cross_entropy` | 正解ラベルCEの係数 | 1 | Q2: 0.2 throughout; Q3/Q4: 0.2, then linear increase to 1.0 | Nonnegative real scalar |
| `brier` | Brier補助損失の係数 | 1 | 0.2 throughout all trained arms | Nonnegative real scalar |
| `gradient_accumulation_steps` | 更新当たりの例数（microbatch 1） | 1 (count) | 4; same examples and reduction across arms | Integer scalar |
| `group_size` | 三値化groupの重み数 | 1 (count) | Initial reference recipe: 128 along each matrix row; retain tail length | Integer scalar |
| `threshold_multiplier` | group平均絶対値に対する閾値倍率 | 1 | Initial reference recipe: 0.7; freeze before outcome inspection | Real scalar |
| `scale` | 非ゼロ三値の絶対値 | 1 (model weights) | Mean absolute retained weight; zero for an all-zero group | Nonnegative real scalar |

For Q3/Q4, updates at `completed_updates` 0–255 use fixed Recovery weights. At 256 the second stage starts with the same weights; at 1,023 it reaches the endpoint exactly. Linear interpolation uses the first and last update of that stage as its endpoints. A smoke-only config uses 8 total updates and 2 Recovery updates; it is never resumed as the 1,024-update experiment.

All loss terms and weights are dimensionless, so their weighted addition is dimensionally consistent. Coefficients are not measurements of actual gradient influence: log raw losses and weighted contributions by modality.

Keep the existing 5e-5 learning-rate candidate, cosine schedule and 3% warmup for the initial comparison, with explicit AdamW betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`, gradient clipping `1.0`, rank 16, alpha 32 and dropout 0.05. No optimizer, adapter or LR restart at the stage boundary. These are research defaults, not a claimed optimum.

## Task 1 — Input and output contract before adaptation

**Files:** Create `src/tiny_omni_decision/input_contract.py`, `tests/test_input_contract.py`; modify `training.py:processor_inputs_for_example` only in the new implementation branch.

**Interface:** `validate_single_media_contract(example: DecisionExample) -> None`.

- [ ] Write tests: empty-media Text passes; one correctly typed media reference passes; Text with media, mismatched media, a second reference, and image-plus-audio all raise an explicit unsupported-input error in the legacy single-media path.
- [ ] Run `python -m pytest tests/test_input_contract.py -q`; confirm the new tests fail before implementation.
- [ ] Add the validator and call it before decoding. Confirm valid historical inputs produce unchanged processor tensors. No audit access and no removal of failed examples from quality denominators.
- [ ] Test distinct answer-token identities and target mapping for 2, 4, 10, 60 and 62 options, including case-sensitive labels and option permutations. Report tie rate/constant predictions for binary Video questions as diagnostics, not proof of a bug.
- [ ] Run the targeted tests plus existing training/decision tests; commit only after they pass. This guard does not implement fusion; it prevents misleading fusion claims.

## Task 2 — A real, versioned schedule contract

**Files:** Create `postquant_policy.py`, `tests/test_postquant_policy.py`, `configs/recovery/decision_respecialization_v2.example.yaml`.

**Interfaces:** `PostQuantConfig` (strict Pydantic model), `LossWeights` (stage, option_kl, cross_entropy, brier), `load_postquant_config(path: Path) -> PostQuantConfig`, `loss_weights_for_update(completed_updates: int, config: PostQuantConfig) -> LossWeights`.

- [ ] Write tests for all fixed values above; reject schema v1, unknown keys, nonfinite/negative weights, omitted budgets, zero-length stages and out-of-range update indices. Require `schema_version: 2` for this new interface; do not reinterpret legacy v1 silently.
- [ ] Confirm tests fail with `python -m pytest tests/test_postquant_policy.py -q`, then implement the pure policy function and strict loader.
- [ ] Assert boundaries 0, 255, 256 and 1,023; monotonic KD decrease/CE increase after 256; Q2 remains fixed; Q3 and Q4 match at every update. Reject unsupported temperature changes.
- [ ] Preserve `configs/recovery/probability_distillation.yaml` as a legacy template. The new runner accepts the new schema only. A config-parse test is not a test that training uses it.
- [ ] Run policy tests and commit. Example config is not a launch manifest: model/data IDs, hashes and resource caps are supplied and validated separately.

## Task 3 — Correct option loss, independent of option count

**Files:** Create `postquant_loss.py`, `tests/test_postquant_loss.py`; preserve historical `decision.py` results.

**Interface:** `option_distillation_loss(student_logits: Tensor, teacher_logits: Tensor, targets: Tensor, valid_options: Tensor, weights: LossWeights) -> dict[str, Tensor]`. Return `total`, `option_kl`, `cross_entropy`, `brier`.

- [ ] Write failing tests for 2/4/10/60/62 options, mixed padded batches, probability normalization, permutation invariance and invalid targets/masks.
- [ ] Implement FP32 log-softmax/softmax over valid choices. Detach teacher logits. Sum categorical KL over active choices per example and then average examples; never divide KL by option count. CE is per-example mean; Brier sums squared probability errors over active choices then averages examples. No sequence-token loss and no full-vocabulary KL.
- [ ] For equal uniform two-choice predictions assert KL is zero, CE is approximately 0.69314718 and Brier is 0.5. Padded logits, even very large ones, must have no effect and receive no gradient. Handle masks without producing zero-times-infinity NaNs.
- [ ] Assert teacher parameters have no gradients; student gradients are finite. Compare accumulation of four single examples with their one-batch loss and gradient, with dropout disabled for the equivalence test.
- [ ] Run `python -m pytest tests/test_postquant_loss.py -q`; commit after red/green verification.

## Task 4 — Frozen Master and correctly keyed teacher cache

**Files:** Create `teacher_cache.py`, `tests/test_teacher_cache.py`; integrate existing split/hash utilities without changing historical artifacts.

**Interface:** `validate_teacher_cache_record(record: dict[str, object], expected: dict[str, object]) -> None`.

- [ ] Write failing tests for changed option text/order/token IDs, target, model/adapter/processor revision, sampled-frame policy and source-media hash. Every mismatch must fail instead of reusing a stale cache.
- [ ] Bind cache rows to sample ID, ordered option text and labels, raw option logits, target index, temperature, preprocessing hash and all relevant model/media hashes. Use teacher eval/no-grad. Hash the full cache manifest and row order.
- [ ] Before quantization, compare Master-plus-Decision-LoRA with the merged Master on a fixed development slice and predeclared dtype tolerances; record deviations and top-1 changes. Block unexplained changes, not harmless rounding by assumption. Reload and hash the merged Master.
- [ ] Cache Q0 signals once on the frozen training examples and unload it before loading the student; do not require two full models in 16 GiB. Keep training/development cache purposes separate and never open a sealed audit.
- [ ] Run `python -m pytest tests/test_teacher_cache.py -q`; commit. Cache changes caused by later clips or combined inputs require new cache identities.

## Task 5 — Reference ternary conversion and feasibility

**Files:** Create `ternary_reference.py`, `tests/test_ternary_reference.py` and an explicit reference-quantizer candidate config.

**Interfaces:** `TernaryTensor` (codes, scales, shape, group size, tail lengths, recipe ID); `quantize_reference(weight: Tensor, *, group_size: int = 128, threshold_multiplier: float = 0.7) -> TernaryTensor`; `dequantize_reference(value: TernaryTensor, *, dtype: torch.dtype) -> Tensor`.

- [ ] Write failing tests for all-zero groups, tail groups, positive/negative values, ties exactly at threshold, nonfinite inputs and exact deterministic roundtrip of codes/metadata.
- [ ] First recipe: per matrix row, form groups of at most 128 actual weights; threshold is 0.7 times group mean absolute weight. Retain weights strictly above threshold in absolute value, give them their original sign and the group's mean absolute retained value as scale; other weights become zero. All-zero groups have zero codes/scale. Compute decisions/scales in FP32; no automatic threshold sweep on validation. This is a simple baseline, not a literature-derived optimum.
- [ ] Resolve exactly the intended decoder linear matrices before conversion. Preserve encoders/projectors/norms/readout and tied-tensor relationships. Save codes/scales separately and install their dequantized values into a distinct frozen reference model; never mutate Q0.
- [ ] Record per-layer zero rates, scale bytes, all non-quantized bytes, adapter bytes and actual resident memory. A code/scale representation stored as ordinary tensors is not claimed to be a packed 1.58-bit artifact. Do not claim speedup from BF16 reference execution.
- [ ] Run `python -m pytest tests/test_ternary_reference.py -q`; then a separately budgeted one-layer forward/backward check and 8-update end-to-end smoke. Require finite outputs/gradients, frozen-base hash stability and save/reload agreement before the 1,024-update launch. CPU tests alone do not establish GPU feasibility.
- [ ] Commit conversion and the recorded smoke outcome separately. If feasibility fails, fix or stop; do not automatically escalate rank, quantizer complexity or paid hardware.

## Task 6 — Matched runner and exact resume

**Files:** Create `postquant_trainer.py`, `tests/test_postquant_trainer.py`, `scripts/run_postquant_comparison.py`. Reuse verified helpers; do not retrofit the active high-precision runner in place.

**Interface:** `run_postquant_experiment(*, arm: str, run_manifest: Path, output_dir: Path, resume_from: Path | None = None) -> dict[str, object]`.

- [ ] Write CPU-toy failing tests showing the runner consumes Task 2 weights at every optimizer update, calls Task 3 loss, validates Task 4 cache keys and uses the correct Q0/Q1 base.
- [ ] Initialize one zero-output LoRA state and copy identical tensors into Q2/Q3/Q4. Keep all non-adapter weights frozen. A quantization-aware LoRA initialization is a later separate ablation, not an unreported advantage for Q3.
- [ ] Freeze run-manifest fields: arm, source commit, config/cache/model/corpus hashes, resolved targets, initial adapter hash, environment/backend, update/sample budgets, fixed evaluation IDs, selector, local GPU-hour/VRAM/disk caps. Refuse missing or inconsistent fields. Do not infer consent to paid compute from this plan.
- [ ] Keep four microbatches on one set of coefficients until the optimizer update. Persist completed update, next stage/local index and coefficients, optimizer/LR state, RNG, exact sampler position, best-checkpoint metadata and prediction IDs. Restore RNG after setup/evaluation.
- [ ] Test uninterrupted versus resumed training immediately before and after the 256 boundary, using a scaled toy schedule. Require identical next input order, weights and CPU deterministic updates. Reject changed hashes or a smoke checkpoint supplied to a longer-budget run; do not reset optimizer at the boundary.
- [ ] Run `python -m pytest tests/test_postquant_trainer.py -q` and the full suite; then run the smoke within its explicit cap. Only a valid frozen run manifest and passing preflight permit the bounded Q2/Q3/Q4 experiment.

## Task 7 — Evaluation, stopping and adoption

**Files:** Create `postquant_comparison.py`, `tests/test_postquant_comparison.py`, `docs/POSTQUANT_EXPERIMENTS.md` when results exist.

**Interface:** `compare_postquant_predictions(arms: dict[str, Path], evaluation_manifest: Path) -> dict[str, object]`.

- [ ] Write failing tests for unequal IDs/order/options/targets, omitted predictions, mismatched preprocessing, and confusion between experiment arms and product variants. No failed item may disappear from the denominator.
- [ ] Compare all arms at the fixed 1,024-update endpoint as the primary bounded experiment. Separately retain development-selected best checkpoints using the same lexicographic rule: higher minimum-modality Accuracy, higher macro Accuracy, lower macro NLL, lower Brier, earlier step. Do not alter the selection rule of the current high-precision run.
- [ ] Report Accuracy/NLL/Brier/ECE by modality, source, task and option count; Video binary majority-class baseline, tie rate and prediction entropy; Audio original and hard-negative results separately. Report development denominators and independent asset counts.
- [ ] Freeze initial research adoption tolerances before launch: Q3's observed minimum-modality gain over Q2 must be at least 2 percentage points; no other modality may lose more than 1 point. These are pilot decision thresholds, not a 95% product claim or universal constants. Report paired asset-cluster intervals and seed variability; noisy or inconsistent differences are UNCERTAIN, not success.
- [ ] Use seed 17 for the first bounded comparison. If promising, repeat the same matched comparison for 19 and 23 within separately approved resource caps; report all seeds, do not select the luckiest. Development intervals remain diagnostics because this data is used in selection.
- [ ] Stop immediately on nonfinite values, base mutation, cache/input mismatch, data leakage, corrupted checkpoints or resource-cap breach; retain failures. No unplanned extra updates until a separate versioned experiment is justified. A valid but inferior Q3 is a negative result, not a broken implementation and not proof that every quantization strategy fails.
- [ ] Adoption and explanation are separate: Q3 need not beat Q4 to be useful at lower deployment cost. Passing engineering tests is not product completion. Keep fixed Recovery if annealing loses. Do not mark any product variant >=95% without its frozen final evaluation.
- [ ] Run `python -m pytest tests/test_postquant_comparison.py -q`, `python -m pytest -q`, and `ruff check .`; commit the report with executed commands and actual results, never expected results.

## Downstream dependency gates — not included in the Q0–Q4 implementation

**Readout:** Preserve 2–62 options and dynamic label order. Verify gathered LM-head rows, bias and any output transformation against the current readout. An equivalent path does not need extra training merely because the projection is smaller; attention replacement is separate.

**True Omni fusion:** Before joint multi-media training or a fusion claim, implement and test all declared combinations with pinned processor/model support, no silently dropped inputs, aligned timestamps where required, and examples needing both sources. The single-media compression pilot remains explicitly limited; this is not approval to replace the final Omni product with separate classifiers. Freeze combined-input evaluation, answerability policy and success criteria before final selection.

**Video:** Train aggregation with video-level labels, not copied labels on every clip. Preserve temporal order and timestamps; compare score-only versus compact-feature aggregation. Keep the original full-video task/split and include equal-total-frame-budget comparisons to separate more observation from better aggregation. Different clip sampling creates a new preprocessing/cache identity. Measure full decode/clip/aggregation latency; chunking does not guarantee less computation. Use non-video replay and genuine combined-input data in final joint specialization.

**Audio:** Keep random-distractor and hard-negative suites separately versioned and speaker/asset disjoint. Verify exactly one correct option and reject mislabeled/ambiguous generated negatives through a prespecified process. Freeze which suites define product quality before final selection; do not retrospectively remove hard examples.

**Deployment:** Benchmark the eventual packed backend against the validated reference and include adapter, encoder, readout, I/O and aggregation costs. Set target hardware, batch, input/frame budgets, p50/p95 latency, memory and startup limits before claiming a lightweight product. Revalidate quality after every backend or re-quantization change.

**Final audit:** Freeze candidate, task definitions, all preprocessing and denominators before a new untouched audit. Require at least 95% in each modality, not an aggregate; report uncertainty and per-source results. Observed audits cannot become a fresh final test for the next tuned candidate. These success targets are not guaranteed by the method.

## Evidence and limits of transfer

- [ParetoQ, arXiv:2502.02631](https://arxiv.org/abs/2502.02631): motivates testing adaptation under extreme quantization; its QAT findings do not prove fixed-ternary-LoRA recovery will succeed.
- [BAM!, ACL 2019, DOI 10.18653/v1/P19-1595](https://aclanthology.org/P19-1595/): supports teacher annealing as a candidate; its BERT results do not establish these coefficients for this Omni model.
- [LoftQ, arXiv:2310.08659](https://arxiv.org/abs/2310.08659): makes adapter initialization a meaningful later ablation; no assumption that a published 4-bit API supports this ternary recipe.
- [PyTorch KL documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html): reduction must match categorical KL; executable tests must run on the pinned installed version rather than assuming latest documentation equals the local backend.

## Completion boundary

A plan file completes planning only. Schedule/loss/cache/quantizer/runner tests complete engineering tasks only. Q0–Q4 results complete a bounded experiment only. Final all-modal 95% plus deployment/fusion gates complete the product. Never substitute one status for another.
