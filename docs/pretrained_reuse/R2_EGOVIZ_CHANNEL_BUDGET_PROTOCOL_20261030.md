# EgoViz fixed-budget channel comparison — protocol

The existing scorer averages the six accelerometer/gyroscope feature channels
before comparing them with video. A small trainable per-feature channel mixer
may retain information lost by that uniform average. Test this hypothesis under
one fixed budget, without changing the dataset, targets, encoders or any prior
checkpoint. The observed development data have already been used repeatedly;
this is exploratory and is not a blind evaluation.

Two readouts start from identical seeded video and sensor projection weights.
The baseline uses the original uniform channel mean. The candidate uses six
softmax-normalized learned weights for each of 512 MOMENT features. Zero gate
logits make the initial candidate mathematically equal to the baseline. This
adds 3,072 parameters to the existing 81,920. The gate is a global channel
pooling rule, not a question-conditioned multimodal gate or LSTM. It adds
capacity as well as changing channel pooling, so any difference cannot by
itself isolate those effects.

Both arms use the same 332 train / 307 development windows, 24 / 20 session
groups, seed 17, four CPU intra-op threads, AdamW LR 0.003, weight decay 0.01, batch size 32 and option
labels. Train each from scratch for **512 complete epochs** with no performance
early stopping; stop on nonfinite loss or gradients. Use the same per-epoch
training order for both arms. Evaluate at epoch 1 and every 8 epochs, record
rolling train CE plus full fixed-weight train and development Accuracy, NLL,
Brier and ECE. Save predictions and checkpoint states at epochs 32, 64, 128,
256, 384 and 512, and a separate best development-NLL checkpoint using min_delta
1e-6. Record every evaluation; do not silently substitute the last checkpoint
for the best. Early results cannot shorten the budget.

At fixed epochs 32, 128, 256 and 512 and at both separately selected best
checkpoints, compare the arms on the identical development IDs with paired
20-session bootstrap (2,000 draws, seed 17); note that selected-checkpoint
intervals do not account for selection. Report candidate-minus-baseline
Accuracy and baseline-minus-candidate NLL/Brier, parameter and checkpoint
sizes, duration, consumed unique/repeated examples, and all curves. Check
option permutation behavior and save/reload parity. The test assesses up to
5,632 updates per arm, not a proof about arbitrarily late grokking or model
capacity ceilings. No sealed data, previously saved weights or prior results
are changed. Do not promote either result on this development comparison.
