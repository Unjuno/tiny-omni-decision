# AVATAR Conditional-Gate Readout Comparison (2026-11-02)

## Question and fixed conditions

The corrected cross-source AVATAR category-match task showed that both frozen
audio and video features matter relative to missing/shuffled controls. ROADMAP
M-2 calls for gated or time-aware feature fusion. This single comparison tests
whether a small option-conditioned gate improves the existing concatenation
MLP on that same development task.

Both readouts use seed 17, the same 11,995 train / 1,655 validation inputs,
the same frozen EfficientAT and VideoPrism caches, AdamW (`lr=1e-3`,
`weight_decay=1e-4`), batch size 128, 50-epoch cap, validation-NLL checkpoint
selection, patience 6, and `min_delta=0.001`. The existing validation was
already used for selection in the baseline readout; these remain exposed
development results. No encoder was updated, no GPU or cloud compute was used,
sealed audit remained unopened, and no model was promoted.

The gate separately projects 384-D audio and 768-D video to 64 dimensions,
computes a sigmoid gate from both candidate-conditioned projections, blends
them per option, then produces one score per option. The operation is shared
across options and is candidate-permutation equivariant. It has 82,177
trainable parameters versus 73,857 for the concatenation MLP (+8,320).

## Results

| Validation readout | Accuracy | NLL | Brier | ECE-15 | Selected epoch |
|---|---:|---:|---:|---:|---:|
| Concatenation MLP | 89.79% | 0.2493 | 0.1429 | 0.0231 | 4 |
| Conditional gate | 89.73% | 0.2664 | 0.1481 | 0.0330 | 4 |
| Video-only MLP | 31.18% | 1.3620 | 0.7408 | 0.0515 | 2 |

The conditional-gate minus concatenation Accuracy difference is −0.06
percentage points, with a paired audio-query-source-group bootstrap 95% CI of
−1.40 to +1.19 points (2,000 replicates, seed 17). This does not support an
Accuracy gain. NLL, Brier and ECE also favor the concatenation baseline.

Gate missing-audio validation Accuracy/NLL are 20.97% / 2.7130; missing-video
is 24.89% / 1.3863; audio-group-shuffled is 23.99% / 5.0236. These controls
show that the fitted gate's scores depend on the supplied streams in this
constructed task, but the gate does not improve on ordinary concatenation.
The validation gate values have mean audio weight 0.509, p10 0.0088, p90
0.9920; 20.5% are below 0.05 and 21.6% above 0.95. Its train/validation NLL
gap is 0.2100 versus 0.1991 for concatenation. The fitted gate frequently
selects one side strongly, but this does not establish that it learned a
general reliability policy.

Macro over 71 validation categories, Accuracy/NLL/Brier/ECE-15 are
87.98% / 0.2965 / 0.1664 / 0.1048 for the gate, versus 89.85% / 0.2574 /
0.1462 / 0.1013 for concatenation. The gate therefore also falls behind on
category-macro metrics. This is one seed and one exposed development task; it
does not establish an architecture-wide negative result.

## Reproduction

Inputs, subset feature caches, configuration, predictions and checkpoints are
in the new output directory:

`C:\CodexArtifacts\pretrained-reuse-r2\avatar-av-decision-20261102-attempt-01\cross-source-option-choices-02\readout-gated-01`

- Training/validation corpus hashes match the corrected cross-source report;
  no train/validation group or asset overlap was found.
- Readout script SHA-256: `82fea515e9bcc8eafbd44f54a6fe401d9a727222db2bb306e5c6c1006d16e13b`.
- Effective config SHA-256: `3cf5a4fa369d4c6a6cd31f7e825e6b36e2094219c58cf3fdceb541cac73a5fd6`.
- Predictions SHA-256: `199cc4cc4b6abb95ee7dac74390b028f86abb00778fda01b91d109bd96407f28`.
- Report SHA-256: `3023bb3032d36571e3e71b6a7d3543a7e4f63a7c185e607b1327f7f8e8844360`.
- Best gate checkpoint SHA-256: `55ee4443ef057237eacf30c8677deffb7110523fed28e37c807847498a33023f`.
- Wall time was 29.37 seconds on CPU; peak CUDA allocation was zero.
- Targeted tests passed (14); full suite passed (2,708 passed, 11 skipped), and
  Ruff passed.

For this task and readout family, the evidence does not justify adding further
gate capacity. Keep the plain concatenation readout as the measured baseline;
keep R2 open for broader task quality and other modalities. A new task should
test temporal/spatial reasoning directly before adding more fusion machinery.
