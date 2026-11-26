# Fixed TRAIN state-dependence diagnostic

The matched native fits completed. The LSTM does not improve fixed-endpoint
temporal Accuracy; its whole-model macro NLL is worse. Before scaling training
or architecture again, ask whether the original native readout actually uses
input state to distinguish different correct answers to the **same question**.
This bounded diagnostic reuses existing feature transport and immutable weights.
The user authorizes autonomous design, local execution and testing; no paid
compute, existing process intervention or new learning is needed.

## Fixed selection, before inference

- Only the common TRAIN corpus and the baseline's complete training ID history.
  Separate consumed6,656 questions from other TRAIN questions. No validation
  examples, calibration or sealed audit. Shared metadata/query banks contain
  other keys, but inference loads only selected TRAIN observations/queries.
- Pair within source/task/consumption stratum, with exactly matching question
  text and the **complete** candidate text set. Different observations and
  different correct candidate texts are mandatory. Preserve all2..40 candidates.
- Sort condition groups and rows by seed17 hashes, greedily match different
  targets without row reuse, rotate condition groups, cap at8pairs per task and
  stratum. Do not change this cap or choose pairs from model performance.
- Use the first row's candidate order and query features for both observations.
  Remap the second row's original target into that complete candidate order.
  Verify corresponding query features match. The only forward input change
  within a pair is the native observation, including its original token layout.
- Preserve unpairable task/stratum counts as missing results, never as passes.
  All512 video TRAIN questions were consumed; unconsumed-video is unavailable.
  The deterministic structural dry-run produces1,250pairs, including8static
  and8temporal consumed-video pairs. This selection is contrast-enriched and
  small per task; it is **not a replacement accuracy benchmark**.

## Models and measurements

Use the attention baseline's original initial weights and its fixed2048 endpoint,
not a checkpoint selected by this diagnostic. No LSTM fitting or model updates.
The196,254,874 full parameter inventory is unchanged; only the889,345-parameter
readout is loaded here, operating on immutable original native feature banks.

For each pair save two full candidate-logit vectors, row/observation IDs, candidate
order and targets. Report individual Accuracy/NLL/Brier/ECE, both-correct rate,
prediction-change rate, total variation of probabilities, and direction of the
two correct-target probabilities. Define the symmetric target effect as:

`0.5 * [(p(target_A|state_A)-p(target_A|state_B)) +
        (p(target_B|state_B)-p(target_B|state_A))]`.

Changing answers alone is insufficient: a model can change to the wrong answer.
Report the effect and both-correct rate separately, per source/task/modality and
consumption stratum. Multiple questions can share assets; counts are not all
independent trials. Do not infer population confidence from these small selected
contrasts. Original broader validation/retention/product gates remain unchanged.

## Execution and safeguards

1. CPU prepare saves the exact manifest and hash before any model inference.
2. Separate GPU smoke takes the first eligible pair from each source, both fixed
   weights; finite outputs, correct input mappings and unchanged model state.
3. Full probe evaluates the entire fixed manifest with both weights.
4. Fresh process repeats all outputs and requires exact array equality.

Require clean committed code, pinned receipts, source/feature hashes, same
environment versions and the previous completed phase's receipt SHA. Never
overwrite a directory. No GPU overlap; stop above2GiB CUDA allocation,8GiB process
peak,20min or below2GiB disk reserve. Save wall time, peak memory, GPU/runtime,
model-state digests and hashes. Scope these costs to readout inference, excluding
the earlier frozen encoder extraction. Existing artifacts remain read-only.

## Interpreting the result

- Unchanged predictions/probabilities under contradictory native inputs support
  insufficient input use by this fitted readout; they do not prove an encoder
  information limit.
- Changed but incorrect answers indicate sensitivity without adequate value
  readout. Consumed/unconsumed contrasts inform coverage, but these strata are
  not randomized and cannot alone prove training-data sufficiency.
- Strong correct state dependence would direct the next work toward broader
  training coverage/generalization. Weak dependence directs it toward the
  existing native-feature/readout path before another temporal architecture.

Choose no hyperparameters or checkpoints from these results. Preserve failures,
report missing conditions, and proceed to one evidence-directed repair with a
separate fixed protocol. This diagnostic does not complete R2 or the95–98% goal.
