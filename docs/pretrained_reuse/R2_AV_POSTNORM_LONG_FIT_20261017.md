# AV post-attention normalization: 32,768-update matched fit

## Result in brief

The opt-in shared post-attention LayerNorm completed all 32,768 optimizer
updates, all scheduled evaluations, and exact best/final checkpoint reload
checks. The fixed selector chose step 16,384. The run used local RTX 3080 Laptop
GPU compute only; it did not load the sealed audit, quantize weights, or change
any existing Teacher/v2 run artifacts.

On the fixed 8,875-row `validation-video` view, Accuracy rose from 38.45% at
step 512 to 46.35% at 2,048, 60.42% at the selected step 16,384, and 61.84% at
the final step 32,768. Video NLL was best at the selected step (1.0865) and
degraded to 1.2687 at the final step. A paired parent-scene bootstrap supports
the Accuracy and NLL improvements versus step 512, with the validation and
checkpoint-selection caveats described below.

The 4-modality `validation-common` macro has a different story: macro Accuracy
rose from 37.42% at step 0 to 46.00% at the final step, but macro NLL worsened
from 1.6302 to 1.6794. The fixed training probe NLL fell to 0.9871. This is a
clear late train/validation divergence on that probe and the validation
warnings are retained. The long fit supports a budget-related improvement on
the dedicated video view; it does not establish a general model limit or make
the candidate a product Teacher.

## Source, run, and integrity

- Source branch: `codex/r2-av-postnorm-stability`
- Source commit: `afe03b78a1d6f997068623a9be6b6cce4ac97386`
- Arm / seed: `E0G0` / 17
- Start / finish: 2026-10-17 18:55:18–23:03:10 JST
- Status: `PASS_MATCHED_AV_TRAIN`; 32,768 optimizer updates, 0 encoder updates
- Initialization: same fitted control8192 E0G0 start tensors, fresh optimizer;
  only the candidate's identity-initialized LayerNorm weight and bias were
  additional model tensors
- Effective config SHA-256:
  `998751914363c623736f418dd2d4d9d54f8d5cbe54de6b195cab4318f6fd5015`
- Plan SHA-256:
  `7dbb60d7569c17a05065640a97367ef0953adacb44d184c82d9f9b6e117d3524`
- Candidate protocol SHA-256:
  `1da068eda87261a46950c76ae210500aa746f6d70cc2729495432e301a582546`
- Fixed input digest:
  `00aa90bed282446e77ae834afc4d9b1d1c1adeedcbaf73e47a49f084c1838868`
- Exact input-audit result SHA-256:
  `69a489b4fa4c81ed155d83328b816409326666f587add8392ea21a4e704efdad`
- Smoke reference SHA-256:
  `c60d597f09a5472a8a5f83dec95f8423048292cfa97c822fc118d5ba92abe67b`
- Sampler/history: seed 17; eight microbatches per optimizer update; the
  existing joint/text/image/audio/video mixture and shuffled option orders
  stayed fixed. Performance early stopping was disabled.
- Sealed audit: `sealed_audit_loaded=false`; no sealed-audit values were read.
- Initialization/order cross-check against interrupted no-norm E0G0: the
  initial model tensors matched exactly except `candidate_norm.weight` and
  `candidate_norm.bias`; Python/PyTorch/CUDA RNG states matched. The first
  5,385 training-history entries had identical sample IDs and option
  presentations. Validation-video IDs and order matched exactly (8,875 rows,
  SHA-256 of newline-joined IDs:
  `0e70eace2c1aade34e329634d424e13a57960dae1e08842290ce224aa9d9ad5a`); the
  validation-common IDs and order also matched (24,346 rows, SHA-256
  `a05f6afbb69b50bb9b3fac46ed95cc98399458ee446c1eeb8cc556c36eccb7db`).

The earlier no-norm run artifact is still interrupted and still says `RUNNING`
in its old metadata. It has 5,385 history entries but its last saved
`result.json` step is 5,376; this stale metadata discrepancy is not rewritten.
Its selected step was 4,096. The comparison below uses only saved predictions
at matching steps and IDs.

## Effective configuration and command

Configuration was E0G0 with one intentional ablation: the shared trainable
LayerNorm after query-conditioned AV attention was enabled. The candidate kept
the existing E0G0 optimizer, sampler, mixture, losses, probe, validation IDs,
evaluation schedule, and selector. It added 384 parameters: 1,732,804 trainable
parameters and 197,098,333 encoder-inclusive parameters before any
quantization. No quantization was run.

Command used:

```powershell
python -m scripts.train_greatest_hits_matched --phase train --arm E0G0 --candidate-output-norm --output C:\CodexArtifacts\pretrained-reuse-r2\greatest-hits-matched-postnorm-train-01 --audit C:\CodexArtifacts\pretrained-reuse-r2\greatest-hits-matched-input-audit-01 --audit-sha256 69a489b4fa4c81ed155d83328b816409326666f587add8392ea21a4e704efdad --reference C:\CodexArtifacts\pretrained-reuse-r2\greatest-hits-matched-postnorm-smoke-03 --reference-sha256 c60d597f09a5472a8a5f83dec95f8423048292cfa97c822fc118d5ba92abe67b --max-seconds 166000
```

The 32,768-update cosine schedule and 62-update warmup were retained. The old
no-norm run was configured for the same maximum budget but stopped after the
5,385th history record; this new run started from the same initial tensors with
a fresh optimizer. It is therefore a matched initialization/data-order
comparison, not a continuation or an optimizer-state resume.

## Learning curve

`validation-common` contains 24,346 examples across audio, image, text, and a
small 400-example video slice. `validation-video` is a separate 8,875-example
audio/video task view. These views are both reported so the larger video
assessment is not confused with the small video slice in the four-modality
common aggregate. The CSV contains common-macro and per-modality
Accuracy/NLL/Brier/ECE, the separate video-view metrics, task Accuracy, and the
fixed training-probe NLL for all rows shown.

| Updates | Common macro Acc. | Common macro NLL | Video view n | Video Acc. | Video NLL | Selector best step |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 37.42% | 1.6302 | 8,875 | 20.44% | 2.4144 | 0 |
| 512 | 36.73% | 1.6547 | 8,875 | 38.45% | 1.6075 | 0 |
| 1,024 | 36.68% | 1.6133 | 8,875 | 42.58% | 1.4620 | 0 |
| 2,048 | 36.65% | 1.6558 | 8,875 | 46.35% | 1.4302 | 0 |
| 4,096 | 37.63% | 1.6130 | 8,875 | 50.68% | 1.2360 | 0 |
| 8,192 | 38.69% | 1.5924 | 8,875 | 45.67% | 1.3221 | 8,192 |
| 12,288 | 42.21% | 1.5147 | 8,875 | 58.10% | 1.2093 | 12,288 |
| 16,384 | 42.86% | 1.5650 | 8,875 | 60.42% | 1.0865 | **16,384** |
| 24,576 | 44.57% | 1.6556 | 8,875 | 61.41% | 1.1789 | 16,384 |
| 32,768 | 46.00% | 1.6794 | 8,875 | 61.84% | 1.2687 | 16,384 |

The selector uses retention eligibility and validation-joint task-macro NLL;
it did not select by video Accuracy. Step 16,384 had validation-joint NLL
1.0381. Step 32,768 had validation-joint NLL 1.1356. The selected checkpoint
and the final checkpoint are separately preserved.

### Per-modality common-view metrics

| Modality / view | Step 0 Acc / NLL / Brier / ECE | Selected 16,384 | Final 32,768 |
|---|---|---|---|
| Audio (n=11,166) | 62.46% / 1.0706 / 0.4655 / 0.0769 | 67.10% / 0.9993 / 0.4206 / 0.0764 | 70.32% / 0.8923 / 0.3727 / 0.0547 |
| Image (n=2,048) | 35.40% / 1.7989 / 0.7353 / 0.0766 | 45.21% / 1.4661 / 0.6315 / 0.0290 | 53.22% / 1.2321 / 0.5575 / 0.0296 |
| Text (n=10,732) | 21.60% / 2.0022 / 0.8378 / 0.0802 | 22.83% / 2.0388 / 0.8353 / 0.0878 | 24.85% / 1.9376 / 0.8236 / 0.1150 |
| Video slice (n=400) | 33.75% / 1.5017 / 0.7556 / 0.1898 | 39.25% / 1.6022 / 0.7734 / 0.2054 | 38.00% / 2.5225 / 0.9034 / 0.3600 |

The more representative video view is the separate 8,875-example validation
view. By source task at steps 512 / 2,048 / 4,096 / 8,192 / 12,288 / 16,384 /
24,576 / 32,768, Accuracy was:

| Video task | 512 | 2,048 | 4,096 | 8,192 | 12,288 | 16,384 | 24,576 | 32,768 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Action | 61.00% | 61.60% | 61.03% | 59.93% | 56.24% | 62.47% | 61.57% | 62.47% |
| Material | 12.02% | 25.76% | 37.17% | 29.48% | 55.17% | 56.78% | 57.56% | 57.89% |
| Reaction | 42.34% | 51.79% | 53.88% | 47.59% | 62.99% | 62.03% | 65.19% | 65.22% |

## Paired video uncertainty diagnostic

Predictions were paired by the exact validation-video question IDs and
cluster-resampled by the 101 `parent` video/scene groups. The procedure used
10,000 bootstrap replicates and NumPy RNG seed 17017; percentile 95% intervals
are reported. The point estimate weights examples, while each replicate samples
parent groups and retains all questions from each sampled group.

| Contrast | Accuracy difference (95% interval) | NLL difference (95% interval) |
|---|---:|---:|
| Step 2,048 − 512 | +7.91 pp [+4.60, +11.63] | −0.177 [−0.248, −0.106] |
| Selected step 16,384 − 512 | +21.97 pp [+15.34, +28.76] | −0.521 [−0.657, −0.378] |
| Final step 32,768 − 512 | +23.39 pp [+16.08, +30.80] | −0.339 [−0.535, −0.131] |

These are development-validation diagnostics on the same view used during
repeated checkpoint evaluation and selection. They are not blind test
confidence intervals and do not account for checkpoint-selection multiplicity.
The apparent 512-to-later improvement is strong within this fixed run, but
the final-step NLL regression relative to step 16,384 shows why Accuracy alone
is not an adequate stopping or selection criterion.

## Matched comparison to the earlier no-norm run

Both runs used the same initial model tensors and exact sample/presentation
order over all 5,385 history entries available in the earlier run. At the same
4,096-update checkpoint, no-norm scored 50.61% / 1.2331 Accuracy/NLL on
validation-video, while postnorm scored 50.68% / 1.2360. The immediate
same-budget quality difference is negligible. At 512, the paired scene-cluster
bootstrap for postnorm minus no-norm Accuracy was −0.33 pp (95% interval
[−1.59, +0.87]) and NLL was +0.0088 (interval [−0.0221, +0.0382]). This does
not show an immediate quality benefit from adding the norm.

The interrupted no-norm history had maximum recorded gradient norm
4,681,888.5 and maximum rolling CE 32,832.9 by its final history entry. The
postnorm long history had maximum gradient norm 713.61, mean gradient norm
3.52, and maximum rolling CE 4.34 across 32,768 updates; all recorded losses
and gradients remained finite, and no NaN/Inf, OOM, or resource stop occurred.
This supports a substantial reduction in the earlier extreme finite-update
spikes under this run, not a proof that the norm alone caused the old run's
behavior. The comparison beyond update 5,385 also combines longer training
with the norm, so it does not isolate the norm as the source of later quality
gains.

## Data coverage and overfitting observations

The complete run logged 262,144 example presentations: 32,768 each for
audio-only, image-only, text-only, and video-only, plus 131,072 audio/video
joint presentations. Accounting reported 108,187 unique questions,
153,957 repeated question presentations, 25,318 unique observations, 12,992
unique media assets, and 11,369 groups. Source presentation counts were
CLEVR-4 32,768; CLEVRER 32,768; Greatest Hits joint 131,072; Lombard 10,926;
STARSS 21,842; and text sources combined 32,768. Repetition is substantial and
is part of the fixed sampler/data plan; it was not hidden or changed.

The `train-common` probe macro NLL decreased from 1.3719 at step 0 to 0.9871
at step 32,768, while `validation-common` macro NLL was 1.6794 at the end,
worse than the initial 1.6302 and selected-step 1.5650. The runner emitted
late warnings for falling train NLL with rising validation NLL and Accuracy
increasing while ECE worsened. On the separate video view, validation NLL
improved through step 16,384, then degraded. This is mixed evidence: learning
budget materially helped video validation, while continued optimization after
the selected point overfit or miscalibrated parts of the broader validation
mix. Rolling microbatch CE is not presented as a fixed-example generalization
gap.

## Hardware, runtime, and artifacts

- Device: NVIDIA GeForce RTX 3080 Laptop GPU; Python 3.11.9; PyTorch
  2.6.0+cu124; CUDA 12.4; float32 deterministic mode; TF32 disabled; 4 CPU
  threads.
- Peak CUDA allocation: 303,542,784 bytes (about 290 MiB); peak process working
  set: 6,624,215,040 bytes (about 6.17 GiB).
- Training time: 5,514.7 s; evaluation/reload time: 9,084.4 s; elapsed wall
  time: about 4 h 7 min 52 s. Cloud use and cost: zero.
- Run directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\greatest-hits-matched-postnorm-train-01`
- Best checkpoint SHA-256:
  `d08120c77ec766427ca5d789629a1f885265f295842784596165c4858ba09b12`
- Final checkpoint SHA-256:
  `176afa85e3acbe37c06962a48d32f95a08f272fd8ce03ff781f20614782b471c`
- Best/final reload checks both passed (`best_final_reload_verified=true`).
- The run directory also contains the effective result/config, complete
  training history, learning curve, per-example evaluation rows, all scheduled
  prediction arrays, `best.pt`, `final.pt`, and numbered checkpoints. These
  large artifacts remain outside Git.
- Compact metrics table:
  [`R2_AV_POSTNORM_LONG_FIT_20261017.csv`](R2_AV_POSTNORM_LONG_FIT_20261017.csv)

## Decision and remaining work

**Budget hypothesis: supported for the fixed video validation view within this
single seed/run.** Video-view Accuracy and NLL improved well past 512 and the
gain was sustained through the selected checkpoint. This does not prove that
training budget was the only cause, nor that the model has reached a limit.

**Product decision: no promotion.** The selected/final models remain far below
the project's multi-modality quality goal; common-view Text is especially weak,
and late Video calibration/NLL degrades. No ternary quantization, compression,
or product artifact was produced. Keep the old E0G0 run, Teacher v1, and other
v2 artifacts unchanged.

Not done in this phase: a new independent/secret evaluation, additional seeds,
comparable v1 evaluation on these exact validation IDs, architecture/parameter
sweeps, quantization, and any product deployment decision. The next useful
validation-only work should examine the late video calibration/NLL divergence
and verify the apparent video gain with a predeclared independent video set
before any compression study.
