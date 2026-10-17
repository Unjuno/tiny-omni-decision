# Conditional AV gate: matched 32,768-update fit

## Result in brief

The E0G1 gated candidate completed its registered 32,768-update schedule,
all validation points, and best/final reload checks. The validation-only
selector chose update 24,576. This is a local development-validation result;
the model is not promoted and the sealed audit was not loaded.

The gate is small in parameter terms, but it did **not** learn a meaningful
dynamic modality weighting policy. Its audio/video sigmoid outputs collapsed
to near zero by the first scheduled point (512 updates); from then through the
selected and final checkpoints, both gates were below 0.01 for every valid
candidate option in the joint AV validation view. The baseline still consumes
all available inputs, and the residual correction receives the query even when
the gated audio/video features are near zero. Therefore the observed joint
score gain does not establish that the gate used complementary AV evidence.

At the fixed 32,768-update comparison, E0G1 versus the matched E0G0 postnorm
control improved joint-view Accuracy by 5.07 percentage points, but the paired
scene-bootstrap interval for joint NLL includes zero. On the video-only view,
E0G1 was 3.83 points lower in Accuracy and had worse NLL. The four-modality
common macro changed very little. This is a trade-off and a gate-collapse
finding, not evidence that the gated fusion mechanism is ready for the product.

## Matched design and integrity

- Arm: `E0G1` (cross-modal exchange disabled, conditional gate enabled) with
  the shared post-attention candidate LayerNorm enabled.
- Control: the completed `E0G0` postnorm run, with the gate disabled.
- Both runs use seed 17, 32,768 updates, the same cosine schedule and warmup,
  optimizer, sampler, source mixture, option order plan, fixed probes,
  validation IDs/order, checkpoint schedule, and selector rule.
- Both effective configs have the same plan SHA-256
  `7dbb60d7569c17a05065640a97367ef0953adacb44d184c82d9f9b6e117d3524`,
  protocol SHA-256
  `1da068eda87261a46950c76ae210500aa746f6d70cc2729495432e301a582546`, and
  input digest
  `00aa90bed282446e77ae834afc4d9b1d1c1adeedcbaf73e47a49f084c1838868`.
- The independent input-audit receipt SHA-256 is
  `69a489b4fa4c81ed155d83328b816409326666f587add8392ea21a4e704efdad`.
- The E0G0 and E0G1 preflight initial model tensors compare exactly; only the
  serialized arm policy differs (`gated=false` versus `gated=true`). Both start
  from the fitted control-8192 state with a fresh optimizer. E0G1 is not an
  optimizer resume.
- The selector is unchanged: retention-eligible checkpoint with the lowest
  validation-joint task-macro NLL, `min_delta=1e-5`, earliest tie. Performance
  early stopping is disabled.
- Sealed audit loaded: false. Ternary quantization: none. Cloud use/cost: zero.
- The only data source is the frozen matched development corpus. The full run
  records 262,144 example presentations; its accounting includes repeated
  questions, so this is not 262,144 unique questions.

## Learning curve

The main table uses the same fixed views for both arms. “Joint” is the
task-macro metric over the 8,875-row AV question view. “Video-only” runs the
same questions through the single-video input path; by design that path bypasses
the AV residual and gate. “Common macro” averages the fixed validation tasks
across audio, image, text, and the 400-row video slice. Full Accuracy/NLL/Brier/
ECE and per-modality values are in the paired CSV.

| Updates | E0G1 joint Acc. | E0G1 joint NLL | E0G1 video-only Acc. | E0G1 video-only NLL | E0G1 common macro Acc. | E0G1 common macro NLL | E0G1 selector best |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20.22% | 2.3507 | 20.44% | 2.4144 | 37.42% | 1.6302 | 0 |
| 512 | 42.42% | 1.4805 | 41.36% | 1.4876 | 36.17% | 1.7101 | 0 |
| 1,024 | 45.35% | 1.4206 | 43.30% | 1.4351 | 35.39% | 1.6540 | 0 |
| 2,048 | 44.22% | 1.5990 | 46.11% | 1.3994 | 36.98% | 1.6758 | 0 |
| 4,096 | 56.04% | 1.1581 | 47.52% | 1.3011 | 38.31% | 1.5829 | 4,096 |
| 8,192 | 54.85% | 1.2099 | 49.21% | 1.2875 | 38.84% | 1.6228 | 4,096 |
| 12,288 | 62.48% | 1.0719 | 51.94% | 1.3063 | 41.13% | 1.5411 | 12,288 |
| 16,384 | 63.90% | 1.0527 | 55.55% | 1.2679 | 41.62% | 1.5359 | 16,384 |
| 24,576 | 67.05% | **0.9895** | 57.86% | 1.3738 | 44.68% | 1.6076 | **24,576** |
| 32,768 | **68.17%** | 1.0374 | 58.01% | 1.4618 | 46.17% | **1.6781** | 24,576 |

Joint Accuracy continues to rise through the final point while joint NLL and
ECE worsen after the selected checkpoint. This is why the fixed selector keeps
step 24,576 rather than selecting the last update. It is not a blind test curve.

### Four-modality common validation at the final update

| Modality (n) | E0G0 Accuracy / NLL / Brier / ECE | E0G1 Accuracy / NLL / Brier / ECE |
|---|---|---|
| Audio (11,166) | 70.32% / 0.8923 / 0.3727 / 0.0547 | 70.45% / 0.8855 / 0.3712 / 0.0623 |
| Image (2,048) | 53.22% / 1.2321 / 0.5575 / 0.0296 | 53.76% / 1.1973 / 0.5455 / 0.0301 |
| Text (10,732) | 24.85% / 1.9376 / 0.8236 / 0.1150 | 24.29% / 1.9233 / 0.8271 / 0.0952 |
| Video slice (400) | 38.00% / 2.5225 / 0.9034 / 0.3600 | 39.00% / 2.5714 / 0.9013 / 0.3585 |
| Macro across modalities | 46.00% / 1.6794 / 0.6704 / 0.1630 | 46.17% / 1.6781 / 0.6680 / 0.1537 |

The 400-row common video slice is not the larger 8,875-row video-only view.
The latter is the view showing a matched Accuracy/NLL regression for E0G1.

## Gate behavior and paired uncertainty

The gate is a sigmoid MLP over the candidate query and pooled AV features. It
multiplies only the added audio/video residual features. The unchanged baseline
still scores the original inputs, and the residual correction also receives
the query directly.

| Update | Mean audio gate (SD) | Mean video gate (SD) | Valid options with both gates < 0.01 |
|---:|---:|---:|---:|
| 0 | 0.5661 (0.0059) | 0.4397 (0.0052) | 0% |
| 512 | 0.00315 (0.00017) | 0.000682 (0.000020) | 100% |
| 4,096 | 0.000805 (0.000054) | 0.000173 (0.000006) | 100% |
| 12,288 | 0.0000834 (0.0000050) | 0.0000124 (0.0000005) | 100% |
| 24,576 selected | 0.0000142 (0.0000009) | 0.00000444 (0.00000020) | 100% |
| 32,768 final | 0.0000145 (0.0000009) | 0.00000513 (0.00000023) | 100% |

The per-option gate standard deviations are tiny and the output remains near
zero for every task type. This is consistent with the gate learning to turn off
its residual path, rather than selecting evidence differently by question.
That interpretation is limited to this saved run; input-intervention inference
is still needed to establish which evidence caused the AV score gains.

Paired uncertainty was computed from saved predictions, pairing exact row IDs
and resampling all questions by their 101 parent video/scene groups (10,000
replicates, NumPy seed 17019, percentile 95% interval). At matched step 32,768:

| View | E0G1 − E0G0 Accuracy difference (95% interval) | E0G1 − E0G0 NLL difference (95% interval) |
|---|---:|---:|
| Joint AV, 8,875 rows | +5.07 pp [+2.10, +7.96] | −0.098 [−0.206, +0.004] |
| Video-only, 8,875 rows | −3.83 pp [−7.26, −0.62] | +0.193 [+0.044, +0.348] |

The selected step comparison and all scheduled matched-step contrasts are in
[`R2_AV_GATED_POSTNORM_PAIRED_BOOTSTRAP_20261018.csv`](R2_AV_GATED_POSTNORM_PAIRED_BOOTSTRAP_20261018.csv).
These intervals describe reused development validation after repeated model
selection; they are not blind-test confidence intervals and do not correct for
selection multiplicity.

## Capacity and local cost

- The gate MLP has 61,634 parameters (960→64→2, sigmoid), about 0.031% of the
  197,098,333-parameter pre-quantization system. It adds about 61,568 linear
  multiply-accumulates per candidate option. Removing the gate entirely would
  leave about 197,036,699 pre-quantization parameters, still within the
  provisional 200M ceiling. No post-quantization size was measured.
- The complete E0G1 readout had 1,732,804 parameters; the run recorded zero
  encoder updates. This count includes the whole candidate readout and should
  not be confused with the gate-only cost.
- Device: NVIDIA GeForce RTX 3080 Laptop GPU; Python 3.11.9; PyTorch
  2.6.0+cu124; CUDA 12.4; NumPy 2.4.6; four CPU threads; float32 deterministic
  mode, TF32 disabled.
- Start/end: 2026-10-18 01:28:04–05:39:33 JST. Wall time 15,088.9 s
  (4 h 11 min); training 5,612.6 s (1 h 34 min); evaluation and reload
  9,202.1 s (2 h 33 min). Peak CUDA allocation 304,802,304 bytes (about
  291 MiB); peak process working set 6,673,784,832 bytes (about 6.22 GiB).
- No cloud compute or cost. Orbitune was not manually stopped; while E0G1 used
  CUDA, process-level inspection showed E0G1 as the only CUDA compute client.

## Artifacts and reproducibility

- Source branch/commit: `codex/r2-av-postnorm-stability` /
  `d3dbe69fa55f8cf866ac67b1a34b27c8dfff605d`.
- Effective config SHA-256:
  `65864faf3ab7f9187d929cc22ff172c4521acbd2e024385272f4de4325570afa`.
- E0G1 smoke reference SHA-256:
  `659745bc6a713494fc733f331c14bf4b7b42432cd5ba8237ea87d792d7112e7d`.
- Best step 24,576; best checkpoint SHA-256:
  `9df53a55e610f4e473dc5f56528edd86610a72683c8f6c40145e9d87d899d87d`.
- Final step 32,768; final checkpoint SHA-256:
  `452f1edc552e5a9129be524ba3f41e55bf2895e0732d8c171bea7ee64a00d0cd`.
- Best and final full-view reload checks passed. The complete model state,
  optimizer states, training history, evaluation rows, learning curve,
  predictions, and receipt remain outside Git at
  `C:\CodexArtifacts\pretrained-reuse-r2\greatest-hits-matched-postnorm-train-E0G1-01`.
- Existing E0G0, Teacher v1, and prior v2 artifacts were read-only and remain
  unchanged. The new compact curve and bootstrap tables are
  [`R2_AV_GATED_POSTNORM_LONG_FIT_20261018.csv`](R2_AV_GATED_POSTNORM_LONG_FIT_20261018.csv)
  and [`R2_AV_GATED_POSTNORM_PAIRED_BOOTSTRAP_20261018.csv`](R2_AV_GATED_POSTNORM_PAIRED_BOOTSTRAP_20261018.csv).

## Decision and next step

The parameter overhead is negligible relative to the 200M pre-quantization
budget, but capacity was not the failure mode observed here. The gate collapsed
to a near-zero residual multiplier by 512 updates and stayed there. The joint
validation Accuracy gain is real on this fixed paired view, but it cannot be
attributed to active gate-based AV evidence use; the query-conditioned
correction path remains a plausible alternative contributor. Meanwhile,
video-only performance regressed.

Do not extend this same gate run or promote its checkpoint. The next useful
step is a no-training, fixed-checkpoint inference diagnostic that separately
ablates the audio/video residual, the query-only correction, and the baseline,
then measures prediction changes under the already-frozen audio/video
replacement conditions. This can determine whether the useful signal comes
from actual cross-modal evidence before changing the gate objective or adding
more training compute. It does not qualify the four-modality 95–98% objective.
