# Postquant numerical core and next worker command

## Status and scope

This branch builds on worker entrypoints commit `1fbe0adad7e9307baabbdc9387b8e27819b78932`.
It implements the numerical-policy v2 loader/schedule, option-only FP32 loss,
tensor-level reference ternary converter, and a standalone CPU checker.
The existing preflight now validates a trained arm's policy contents against
its launch arm/profile/budget, not only its file hash.

**It does not implement or launch the Gemma postquant trainer.** Full-model
conversion/merge, teacher-cache generation, data/fusion validation, model save/reload,
optimizer/RNG/sampler exact resume, resource-limited GPU execution and packed kernels
remain downstream. No `postquant_trainer.py` or runnable-backend capability flag is added.
The 8-update model smoke and Q2/Q3/Q4 experiments remain blocked, not passed.
Existing Teacher code/config/weights and legacy Recovery YAML are unchanged.

## Worker command (Windows or Linux, from repository root)

Inspect the policy with no torch import and no writes:

```shell
python scripts/check_postquant_numerics.py --policy configs/recovery/decision_respecialization_v2.example.yaml --completed-updates 256
```

Add an explicit check of tiny synthetic CPU tensors only:

```shell
python scripts/check_postquant_numerics.py --policy configs/recovery/decision_respecialization_v2.example.yaml --check-numerics
```

The latter computes a two-choice loss and one backward pass, plus a four-weight
reference roundtrip. It performs **zero optimizer updates**, loads no model/corpus,
uses no GPU, and makes no network request. It is a numerical plumbing check, not
an Accuracy/latency benchmark or the real-model 8-update smoke. The script has no
`--execute` option. JSON is printed to stdout; errors return exit code 2.

For equal uniform two-choice probabilities: KL is 0, CE is approximately
0.69314718, and summed Brier is 0.5. These are analytic fixture expectations,
not measured model performance.

## Numerical contract

| Field / symbol | Meaning (日本語) | SI unit | Definition / conditions | Type |
|---|---|---|---|---|
| `completed_updates` | 次の更新前の完了optimizer更新数 | 1 | Integer from 0 to total minus 1; not microbatch index | Scalar integer |
| `total_updates` | 更新予算 | 1 | 1024 for comparison; 8 for standalone smoke profile | Scalar integer |
| `recovery_updates` | 第一段階の更新数 | 1 | Q3/Q4: 256 or 2; Q2: 0 (fixed policy) | Scalar integer |
| `option_kl` | 教師選択肢分布へのKL係数 | 1 | Initial 1.0; staged endpoint 0.2 | Nonnegative real scalar |
| `cross_entropy` | 正解ラベルへのCE係数 | 1 | Initial 0.2; staged endpoint 1.0 | Nonnegative real scalar |
| `brier` | Brier損失係数 | 1 | 0.2 throughout initial candidate | Nonnegative real scalar |
| `temperature` | 蒸留温度 | 1 | Exactly 1.0; other values rejected in this version | Real scalar |
| `student_logits`, `teacher_logits` | 学生・教師の選択肢スコア | 1 | Nonempty batch by 2..62 padded choices; valid entries finite | Real matrices |
| `valid_options` | 有効な選択肢の印 | 1 | Same shape as logits; at least two true values per row | Boolean matrix |
| `targets` | 正解の選択肢index | 1 | One int64 valid index per example | Integer vector |
| `weight`, `scale` | 三値化元の重みとgroup尺度 | 1 (model units) | Dense real matrix; FP32 assignment; nonnegative finite scale | Matrix / group scalars |
| `group_size` | 一行内の重みgroup長 | 1 | Initial 128, last group uses actual tail only | Positive integer |
| `threshold_multiplier` | group平均絶対値の閾値倍率 | 1 | Initial 0.7, retained comparison is strictly greater | Nonnegative real scalar |

All loss terms and coefficients are dimensionless, so addition is dimensionally
consistent. KL is summed over active choices then averaged over examples; never
average it over padded/active choice count. Brier also sums over choices before
averaging examples. CE averages examples. Padded values are removed before
normalization and have zero gradient; teacher logits are detached. Computation
uses FP32 even with BF16 input, including under autocast.

For Q3/Q4 the first 256 updates use Recovery coefficients. Update index 256
starts stage two with the same weights; index 1023 reaches the exact endpoint.
The same mapping holds for smoke indices 2 and 7. No optimizer restart or LR
schedule behavior is implied by this pure coefficient function. Test coverage
shows the coefficients reach the actual numerical loss/gradient; it does not
prove integration into the not-yet-implemented Gemma trainer.

The strict loader rejects unknown fields, duplicate YAML keys, aliases, wrong
schemas/types/budgets, nonfinite/negative coefficients, unsupported temperature
and protected paths. Launch v1 and numerical policy v2 are different schemas.
Q0/Q1 remain evaluation-only and their launch preflight remains hash-level;
no numerical training policy for those arms is claimed.

Q2 must specify fixed schedule, zero recovery updates and identical start/end
coefficients. Q4 uses the Q3 schedule with its own arm identifier. Every change
requires a new policy hash in its launch manifest; no silent overrides.

## Ternary reference boundary

`quantize_reference` accepts a matrix, not a model. It never modifies that
matrix in place. Each group uses mean absolute weight times 0.7 as the initial
threshold and mean absolute retained weight as scale. Ties become zero; all-zero
groups retain zero scale. The matrix has ordinary int8 codes and FP32 scales;
its `storage_report` explicitly excludes metadata serialization and other model
components and reports `packed: false`. This is not 1.58-bit packing or a fast
kernel. Mapping actual decoder weights, preserving ties and measuring full
model/adapter bytes still require the planned model-conversion implementation.

State conversion uses tensors and primitive metadata only. The unit test
roundtrip uses `torch.load(..., weights_only=True)` on an in-memory synthetic
buffer; the CLI never deserializes an external checkpoint.

## Verification and research

Local verification uses Python 3.13.5 / CPU PyTorch 2.10.0 / Pydantic 2.13.4 /
pytest 9.0.2 in an isolated partial checkout. The project supports Python
3.11/3.12 and pins ML PyTorch 2.6, so these local tests alone do not verify the
target environment. A CPU-only PR workflow tests Python 3.11 and torch 2.6.0
on Linux and Windows. Consult the actual CI outcome; a workflow definition is
not evidence it passed. No model download occurs in these jobs.

- [PyTorch KL reduction](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.kl_div.html):
  `mean` divides by all elements; use categorical sum then batch mean.
- [Pydantic strict validation](https://docs.pydantic.dev/latest/concepts/strict_mode/):
  reject coercion and unknown keys rather than pretending the run contract matches.
- [PyTorch checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html):
  reloading model weights alone is not optimizer/sampler/RNG exact resume.

These sources informed implementation semantics, not claims of 95% Accuracy,
optimal coefficients, ternary sparsity benefits or deployment speed. Numerical
correctness, statistical quality, and runtime efficiency remain separate checks.
