# Fixed parent probability retention: measured comparison

[Frozen protocol](R2_BINDING_RETENTION_KL_PROTOCOL_20261015.md) ·
[Machine-readable evidence](R2_BINDING_RETENTION_KL_RESULTS_20261015.json) ·
[Preserved CE-only control](R2_CONDITION_BINDING_RESULTS_20261015.md).

## Outcome

One fixed seed-17 run completed all 512 updates. Step 512 was the **only trained
checkpoint passing the unchanged retention gates**, and is selected. It reaches
100% on 1,024 narrow semantic-binding validation questions, including all four
questions per state and all condition-change pairs. Preserve this as a research
candidate; it is not a product Teacher or native four-modality quality result.

The original-task probability degradation from the CE-only control is reduced.
Retention is not exact: curriculum NLL increases by 0.017793 versus the parent,
and extremum Accuracy loses 0.9765625 percentage points, near the fixed one-point
limit. The curriculum NLL bootstrap interval is entirely positive and extends
past the 0.02 gate. Thus a point-estimate gate pass is not proof of robust
non-inferiority. No threshold was changed after observing the run.

Fresh-process inference exactly reproduces all four selected score arrays and
all 12 selected reload payloads. Transfer evaluation and sealed audit were not
run. The next decision is independent conditional/native transfer with frozen
weights, not further coefficient, learning-rate or budget tuning here.

## Controlled change

Both fits start from the same retained 22,713,601-parameter coverage parent.
Same 4 binding + 4 original replay questions/update, 512 AdamW updates,
LR 2e-5 constant, weight decay 0.01, clip 1, seed 17, FP32 SDPA, TF32 off,
dropout disabled and four CPU threads. The same 10,647,169 encoder/classifier
parameters train; embeddings/pooler remain frozen. No architecture, data,
option-count, schedule or selection change.

Control: `0.5 * binding_CE + 0.5 * replay_CE`.
Candidate: `0.5 * binding_CE + 0.5 * (replay_CE + KL(parent || student))`.
Temperature 1 and coefficient 1 were fixed before the run. Every term averages
over questions; each keeps all its actual candidates without padding mass.
Parent probabilities are detached.

The regularizer adds replay strength as well as probability targets; there is
no loss-weight-matched CE control. This is output retention from the existing
parent, not new general teacher knowledge or embedding imitation. No second
parent model is resident: cache outputs before updating the same loaded model.

Parent targets use only the 2,048 original TRAIN replay rows, in the actual 512
four-question sampler batches. Cache generation presents 7,132 candidate pairs
in 19.877 seconds; every entry hashes the complete row including candidate order.
Initial training KL is 0.0. This inference is separate from optimizer consumption.
Step 0 scores exactly match all four control arrays (binding, original,
curriculum and fixed TRAIN probe). Neither validation nor transfer targets enter
training or the parent cache.

## Data and interpretation boundaries

Immutable corpus: `condition-binding-corpus-01`, result SHA-256
`957e2ce3ade65c5d2caa26a1cbb0e95a3adeee5cf5b312110f8c2a7e3cfed971`.
Binding TRAIN: 2,048 questions / 512 states; replay: 2,048 questions / 1,815 states.
Both the complete run and selected checkpoint consume 4,096 unique question IDs,
zero repeated optimization IDs, and 2,327 source-qualified states. Source/task
counts are retained in every curve entry. Cache inference is not extra unique data.

The 8,424 upstream TRAIN images contain 5,636 distinct state combinations; the
256 binding selector states exclude all those combinations and previous diagnostic
states. These are annotation-derived semantic states, with no media consumed here.
Binding validation has 1,024 questions and the same 40 typed choices. All 2,540
old original questions / 732 states and 2,048 curriculum questions / 256 states
remain in validation. Curriculum TRAIN remains absent from replay, as in control.

Observed validation is reused for development and retention checks. The old
transfer files are checksummed only to verify the unchanged corpus; none are
parsed/scored. No sealed audit is accessed. Narrow binding success cannot establish
native image/audio/video, unfamiliar-language or OOD product quality. No new
language templates were selected from this result.

## Learning curves and fixed selection

Eligibility: every old task Accuracy decrease <=1 pp and each old source NLL,
Brier and ECE increase <=0.02, 0.01 and 0.02 respectively. Among eligible points,
select minimum binding validation NLL, earliest tie; step 0 is the fallback.
No quality-based early stop. Best and final weights are stored separately.

| Step | Binding Acc / NLL | Original Acc / NLL | Curriculum Acc / NLL | Eligible |
|---:|---:|---:|---:|---|
| 0 | 31.74% / 2.189042 | 63.58% / 0.835591 | 84.08% / 0.375370 | True |
| 128 | 100.00% / 0.001550 | 64.25% / 0.840417 | 83.79% / 0.405235 | False |
| 256 | 100.00% / 0.001326 | 63.82% / 0.848664 | 84.38% / 0.416145 | False |
| 384 | 100.00% / 0.001791 | 64.09% / 0.840955 | 83.84% / 0.406564 | False |
| 512 | 100.00% / 0.002140 | 63.86% / 0.837478 | 83.94% / 0.393162 | True |

At 128, failures include original ECE and curriculum extremum Accuracy/NLL/Brier;
at 256, original noul Accuracy/ECE and curriculum NLL; at 384, curriculum extremum
Accuracy/NLL. All pass at 512. This is selection by the predeclared rule, not an
automatic last-checkpoint policy. Full per-task values and gate failures are in JSON.
Fixed TRAIN Accuracy is 100% at 512, NLL 0.000400, versus validation 100%, NLL
0.002140. There is no Accuracy gap on this narrow task, but language/native
transfer is untested. Rolling optimization losses are not fixed-example gaps.

| Validation / model | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| binding / parent | 31.74% | 2.189042 | 0.944230 | 0.328490 |
| binding / CE-only 512 | 100.00% | 0.000800 | 0.000124 | 0.000767 |
| binding / KL 512 selected | 100.00% | 0.002140 | 0.000501 | 0.001998 |
| original / parent | 63.58% | 0.835591 | 0.470606 | 0.018915 |
| original / CE-only 512 | 63.54% | 0.898312 | 0.485632 | 0.097916 |
| original / KL 512 selected | 63.86% | 0.837478 | 0.469379 | 0.026582 |
| curriculum / parent | 84.08% | 0.375370 | 0.214324 | 0.016823 |
| curriculum / CE-only 512 | 84.23% | 0.437499 | 0.228650 | 0.043234 |
| curriculum / KL 512 selected | 83.94% | 0.393162 | 0.222340 | 0.023702 |

The new model trades slightly higher binding NLL than the CE-only final for better
old-task probability retention. It does not improve every metric over the parent.
Original choice/noul/score Accuracy at 512 is 62.30% / 73.56% / 49.17%; curriculum
conditional-selection/count/extremum/temporal-order is 71.48% / 59.77% / 88.67% /
92.77%. These text-state tasks are not native video or audio measurements.

| Comparison / validation | Accuracy difference, pp [95% interval] | NLL difference [95% interval] |
|---|---:|---:|
| selected_vs_parent / binding | +68.262 [+66.797, +69.727] | -2.186902 [-2.250880, -2.120380] |
| selected_vs_parent / original | +0.276 [-0.946, +1.480] | +0.001887 [-0.008100, +0.012014] |
| selected_vs_parent / curriculum | -0.146 [-1.074, +0.781] | +0.017793 [+0.005303, +0.030742] |
| final_vs_control_final / binding | +0.000 [+0.000, +0.000] | +0.001340 [+0.000629, +0.002247] |
| final_vs_control_final / original | +0.315 [-0.672, +1.333] | -0.060834 [-0.076798, -0.045388] |
| final_vs_control_final / curriculum | -0.293 [-1.025, +0.439] | -0.044337 [-0.061142, -0.029014] |

NLL uses stable FP64 log-softmax; Brier sums classes; ECE uses 15 bins. Paired
differences resample complete states 10,000 times, seed 17, percentile 95% with
linear quantiles. Original groups have 1–6 questions, so each replicate divides
resampled sums by the resampled question count. Binding groups keep four questions
together and curriculum groups eight. These are single-seed development intervals
with checkpoint-selection effects and no multiplicity correction, not blind audit
or multi-seed evidence. Selected and final happen to coincide here.

## Execution, reproducibility and cost

Run source commit: `b5b12f095c158ca4453ef1c8fa87b2e75c8e0747`.
Base: `cross-encoder/ms-marco-MiniLM-L6-v2`, revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`. Parent delta SHA-256:
`b063f124e59c8a4c1b09c7bf2a8423846f73489d7ce8cd6649735745bc6cebf5`.
Optimizer reset; dense encoder/classifier deltas contain no optimizer/scheduler/RNG
state and are not exact training resumes.

RTX 3080 Laptop 16 GiB, Windows/Python 3.11.9, PyTorch 2.6.0+cu124 / CUDA 12.4,
Transformers 5.6.2, NumPy 2.4.6, tokenizers 0.22.2, safetensors 0.8.0, psutil 7.2.2.
No environment upgrade or paid compute.

| Stage | Updates | Process-body seconds | Peak allocated VRAM bytes | Peak host bytes |
|---|---:|---:|---:|---:|
| TRAIN-only smoke | 2 | 8.929 | 1,721,917,440 | 1,586,393,088 |
| Primary | 512 | 470.573 | 2,055,053,312 | 1,652,940,800 |
| Fresh selected replay | 0 | 52.903 | 306,067,456 | 1,298,239,488 |

Total recorded process-body time: 532.405 seconds (8.87 minutes); not GPU-busy
seconds. Primary started 2026-10-15 04:25:48.503735 UTC and ended 04:33:39.074893 UTC
on normal budget completion. Its update bodies total 130.413 seconds, scheduled
validation 258.358 seconds, parent cache 19.877 seconds; loading/reload/accounting
make up the remaining time. Peak reserved VRAM is 2,703,228,928 bytes. No numerical,
freeze or resource failure. Pre-clipping gradient norm range is 0.21445–48.09635;
KL range 0–0.326752. Frozen and all pinned source/code hashes remain exact.

Primary counted 509,698 candidate-pair forwards across cache/training/validation/
reload. It truncates 2,124 pair presentations across 163 old question IDs with
unchanged `only_second` handling, actual maximum input 512 tokens. The tokenizer's
raw-length warning precedes that truncation; it is not an oversized model input.
Do not compare total primary wall time to the old run as a training speedup:
this protocol omits the old transfer evaluations, which changed total workload.

Commands from `C:/CodexWorktrees/pretrained-reuse-omni`; each run rejects an
existing output directory:

```powershell
python -m scripts.train_condition_binding_retention --smoke --output C:/CodexArtifacts/pretrained-reuse-r2/condition-binding-retention-smoke-01
python -m scripts.train_condition_binding_retention --output C:/CodexArtifacts/pretrained-reuse-r2/condition-binding-retention-fit-01
python -m scripts.train_condition_binding_retention --output C:/CodexArtifacts/pretrained-reuse-r2/condition-binding-retention-replay-01 --replay-reference C:/CodexArtifacts/pretrained-reuse-r2/condition-binding-retention-fit-01 --replay-sha256 040c4e838af84f2a19ea57ffde894189891383adb0df396199e6e7c62aade82c
python -m scripts.summarize_binding_retention --root C:/CodexArtifacts/pretrained-reuse-r2 --primary-sha256 040c4e838af84f2a19ea57ffde894189891383adb0df396199e6e7c62aade82c --replay-sha256 b1b3ca77202ec826c44049cada921f9f0daff5f1e33986cc719930fcd9c93296 --output C:/CodexArtifacts/pretrained-reuse-r2/condition-binding-retention-publication-01.json
```

| Artifact | SHA-256 |
|---|---|
| primary result | `040c4e838af84f2a19ea57ffde894189891383adb0df396199e6e7c62aade82c` |
| fresh selected replay result | `b1b3ca77202ec826c44049cada921f9f0daff5f1e33986cc719930fcd9c93296` |
| local publication | `0ac9e8a26285b3a6500b72d98ea98b6d5950b9885f206484b899e5cb78175f62` |
| best-delta.pt | `c87ddf2197e276814e4941efb7ad7431d416316eb786ea59b8331925ca16ab02` |
| final-delta.pt | `89d96615b6035f509c310880781cb888d843184f3c846df3b4fc4d8ca573e743` |
| effective-config.json | `3fbfef96bcc439b5e1565c6a78c2191f77e05fc314db2ca1f28d5e3267c10a10` |
| teacher-cache.json | `818e2036050c13cb73337b7fb65bb9e09a237179ce3b71ef62e1ef7321c03a96` |

The first publication invocation supplied the replay metadata hash read before
its process completed; the hash gate rejected it and created no publication.
After exit 0, the final replay result was hashed and publication passed without
weakening any check. No additional fit or evaluation was run for that correction.
The tracked JSON preserves the local publication content; Git newline handling
can make working-copy byte hashes differ. Weights/cache/media stay outside Git.

Ruff passes; full CPU 1,589 passed / 1 skipped; simulated no-Torch 1,160 passed /
111 skipped; all 25 manifest/catalog validation commands pass. CPU loss/cache and
grouped-bootstrap tests pass. Read-only review checked the analytic student
gradient, all 2,048 cache positions, unchanged selector/defaults, exact baseline
parity, gate reconstruction and replay guards; no blocker found.

All previous artifacts, Teachers, frozen corpora, worktrees and audit sets remain
protected. New candidate deltas have separate output paths. No main merge,
cloud use, extra seed, quantization or product promotion.

## Next action

Freeze this candidate and parent for one independently specified transfer
comparison: unfamiliar conditions/candidate descriptions/combinations, followed
by matched oracle-state versus perceptual-state inputs. Keep all hard cases,
question counts, control conditions and acceptance rules fixed before inference.
The already observed transfer set is not a new blind benchmark. Do not search
more KL coefficients, prompts or steps against the current selector.

If oracle transfer fails, improve conditional semantics and diverse supervision.
If oracle transfers but perceptual inputs fail, investigate grounded information
loss. Native temporal and genuinely complementary joint-input quality remain
separate open gates. The <=200M four-modality 95–98% product goal is not complete.
