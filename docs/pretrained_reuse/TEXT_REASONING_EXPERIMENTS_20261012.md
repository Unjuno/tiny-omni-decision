# Completed conditional-reasoning experiment ledger

This ledger consolidates completed runs after the original epoch3 text adaptation.
It is not a model leaderboard: rows differ in budget, trainable scope and data.
All share the same22.7M base model; no added model parameters. Full history, source
commits and failures remain in individual reports. All use seed17; no seed search.

| Run | Updates | Trainable | Selected step | Selected original Acc | Selected curriculum Acc | Timed seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conditional-repair-pilot-01 | 512 | 1774849 | 512 | 0.57835 | 0.24365 | 253.6 |
| conditional-repair-full-coverage-01 | 2048 | 1774849 | 768 | 0.57953 | 0.23291 | 884.7 |
| conditional-repair-encoder-coverage-01 | 2048 | 10647169 | 2048 | 0.60236 | 0.80566 | 1141.0 |
| conditional-repair-numeric-range-01 | 2048 | 10647169 | 2048 | 0.61024 | 0.24512 | 1119.3 |
| conditional-repair-range-warmstart-01 | 2048 | 10647169 | 1408 | 0.60315 | 0.82227 | 1122.3 |
| conditional-repair-lexical-range-01 | 2048 | 10647169 | 1408 | 0.60433 | 0.82959 | 1123.8 |

## Findings that constrain the next action

- Final-block-only training did not acquire useful numeric decisions despite
  complete unique curriculum coverage. Increasing budget alone was insufficient
  in that measured setup, not a universal capacity ceiling.
- Updating all six existing blocks acquired narrow-format behavior (80.57%),
  while the fixed256-question format-transfer diagnostic remained33.59%.
- Controlled perturbations identified numeric-range and wording fragility;
  identifier/field-name changes had much smaller effects.
- Numeric-range augmentation at the same budget failed to fit its own training
  slice (27.25%) and failed transfer (24.27%). Improved calibration mainly reduced
  overconfidence; it did not establish useful reasoning.
- Contextual numeric tokenization changed from all single-token values to32.79%
  two-token values. Magnitude and vocabulary also changed: this is a clue, not a
  causal diagnosis or proof a larger model is needed.

The learned-initialization control completed; final expanded-range training fit83.50%,
selected narrow validation82.23%, with worsened original-text NLL/calibration.
It adds prior training exposure; do not describe it as equal-total-compute.
No final audit, modality-wide95% result, compression or product promotion exists.
Text is one unresolved prerequisite; image/audio/video general reasoning and
joint fusion remain open. Individual task-specific image/audio successes cannot
substitute for those gates. See ROADMAP for the complete R0遯ｶ霑ｭ8 objective.

## Accounting limits

Times exclude initial loading/tokenization and are not power/cost measurements.
Unique questions are per-run only; the runs reuse IDs, states, inherited weights
and development validation. Do not sum them into lifetime unique coverage.
Weights-only artifacts do not support strict optimizer/RNG resume. All stages
retain separate outputs, histories, selected/final weights and source hashes.
