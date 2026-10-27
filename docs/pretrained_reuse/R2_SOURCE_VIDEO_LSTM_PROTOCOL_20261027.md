# Source-candidate visual LSTM: fixed train-only CV protocol

Test whether chronological visual candidate features improve the existing
joint readout on the same three fixed video-group folds. Use only the 2,593
official-train records and already cached VideoPrism ROI frame features and
pooled EfficientAT audio; no old development features/labels, official
validation, test or sealed audit. Retain every record and candidate.

Preserve the existing candidate mean/last-minus-first and audio input path.
Add one small LSTM over each candidate's actual ordered ROI frames. Add its
last-minus-first hidden state as a learned residual to the original visual
projection, with zero-initialized residual projection. For one-frame
candidates the added delta is exactly zero. Mask padded candidates and pack
variable frame sequences; never duplicate or interpolate frames. Initialize the
original readout first with the same seed so its starting weights and logits
match the baseline. The audio sequence remains pooled: this experiment tests
visual temporal coverage, not audio temporal learning or synchronization.

Train only the joint variant: seed 17, exactly 7 epochs, batch 32, AdamW
0.001/weight-decay 0.0001 and gradient clipping 5, same train groups and
permutation stream as the existing joint baseline. No checkpoint selection or
early stopping. Save three independent fold checkpoints and all held-out
predictions; evaluate and compare paired Accuracy/NLL/Brier/ECE with the saved
baseline using video-group bootstrap. Report total/trainable parameters and
runtime. This adds capacity and visual sequence information at once; it cannot
isolate either effect. The same OOF validation data have been observed, and
source labels/participants, clocks and oracle tracks remain limitations.
Do not present this as blind testing, product quality or a 95% result.
