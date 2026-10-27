# Fixed visual-LSTM checkpoints: frame-order intervention

On the three saved out-of-fold visual-LSTM checkpoints, replay the original
predictions and require saved-probability, identity and metric parity. Then
reverse only each candidate's actual valid ROI frame sequence presented to the
LSTM. Keep the original visual mean/delta path, audio features, target, candidate
order, weights and masks unchanged. Do not reverse padding, duplicate frames,
or rerun training. The residual is identically zero for one-frame candidates.

Use all 2,593 original training-source records from the same 320 video groups.
Report paired Accuracy and NLL differences and 10,000-draw seed-17 video-group
bootstrap intervals. Also report counts of examples with any multi-frame
candidate and, for visible-source examples, at least one multi-frame correct
candidate. These are conditional diagnostics, not a selected test set.
No old development/official validation/test labels or sealed audit are opened.

This probes reliance on the residual sequence order while the base path keeps
forward-order information; it is not a full physical time reversal. Feature
timing and oracle-source correctness are not established. The subset, OOF
models and source have been observed, with one seed and shared training folds.
The result is exploratory and cannot prove temporal understanding or establish
product quality.
