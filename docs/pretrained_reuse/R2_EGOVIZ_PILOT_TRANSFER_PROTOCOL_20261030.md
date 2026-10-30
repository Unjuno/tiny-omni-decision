# Fixed EgoViz model transfer to prior pilot — protocol

Before seeing transfer outcomes, evaluate the already selected 24-session
model and its fixed training-out-of-fold inverse temperature 3.5672662302704614
on the prior pilot's separate 16 capture sessions. These sessions were excluded
from the 24-session training and 20-session development plan. Verify zero
overlap in capture-session IDs, source chunk IDs and sample IDs against both
new splits. Hash-check all frozen inputs, checkpoints and calibration metadata.

Report the prior pilot's eight train sessions / 106 examples and eight
development sessions / 110 examples separately, with Accuracy, NLL, Brier,
ECE and mean confidence before and after fixed scaling. Do not fit, calibrate,
change a checkpoint or select a scalar using these records. Save IDs,
targets, per-option logits/probabilities and actual command in a new output.
Check ranking preservation and a fresh-process probability reload.

For context, evaluate the prior pilot's selected joint model on its own
previously observed 110-example development set and reproduce its saved
metrics. Compare the two fixed models on the same examples with paired
eight-session bootstrap (2,000 resamples, seed 17) for Accuracy and NLL.
Also report new-model original versus scaled NLL/Brier on those same examples.
The old model was selected on that set; its pilot training split is not an
independent comparison and is not used for an old-model performance claim.

Both pilot splits and the old model's development metrics have been examined
before. The architecture and task were previously developed using this pilot,
and a capture-session split does not prove participant or site independence.
This is an **unseen-session transfer check for the new weights**, not a blind
audit or independent evidence for the whole architecture. Pilot MOMENT features
were computed on CUDA, whereas the new training features were computed on CPU;
checkpoint and preprocessing fingerprints match but backend numerical effects
are not measured. No fresh model fitting, retraining, changes to existing
artifacts or sealed media access are authorized in this evaluation.
