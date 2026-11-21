# Capacity-matched nonrecurrent control for native temporal readout

## Question

The existing 2,048-update comparison found a small Video NLL/Brier difference
for the LSTM arm, but its readout also had 1,036,416 more trainable parameters
than baseline. This one-arm follow-up holds the original Path A corpus, split,
seed, optimizer, sampler, initialization, evaluation points and selector fixed,
and replaces only the recurrent cells with tokenwise feed-forward blocks. The
development validation has already been observed in the earlier baseline/LSTM
comparison; this experiment is a capacity diagnostic on that same exposed
validation, not independent generalization evidence or a product selection.

## Frozen conditions

- Seed 17; the original 889,345-parameter native readout initialization.
- Same frozen native common corpus, query cache, row order, option shuffling,
  TRAIN probe, all 24,346 development-validation questions and exact metrics.
- 2,048 optimizer updates; four single-question microbatches/update; AdamW
  `lr=0.001`, 62-update warmup, cosine decay, weight decay `0.01`, gradient
  clipping `1.0`; evaluate every 256 updates.
- Same selector: equal-task/source/modality macro NLL, `min_delta=1e-5`, earliest
  checkpoint on ties. Preserve best and final checkpoints separately.
- No encoder updates, new data, calibration, hyperparameter sweep, sealed audit,
  or product promotion.

## Control architecture

For speech and video tokens only, apply the same `D -> 192` projection and
LayerNorm, then a tokenwise `192 -> 768 -> 192` GELU MLP and a zero-initialized
`192 -> D` residual projection. Each token is transformed independently; no
token can read another token or carry state. Video spatial/time coordinates are
still validated with the LSTM arm's fixed-grid layout check. The common
candidate readout remains unchanged. The zero residual projection makes the
initial function exactly match the shared baseline readout.

The control adds 1,035,264 parameters versus the LSTM's 1,036,416: a difference
of 1,152 parameters (0.111% of the LSTM addition). This nearly isolates
cross-token recurrent mixing from added trainable capacity; it does not match
FLOPs or prove a universal architecture result.

## Execution decision

Proceed only if the source and external corpus hashes pass, GPU admission finds
the local device idle, disk reserve is at least 2 GiB, a TRAIN-only smoke is
finite and save/reload exact, and its full-run projection is within the existing
2-hour cap. Run one seed-17 control fit only. Require a fresh-process replay of
all scheduled checkpoints to match saved predictions exactly. A failure or
resource stop is incomplete, not evidence against recurrence.

Interpret the control against the already saved baseline and LSTM predictions
as a fixed-experiment comparison. Report all modalities, task-level measures,
train-probe curves, uncertainty and resource cost. Because validation was
previously observed and the comparison is one seed, no candidate is promoted
from this result.
