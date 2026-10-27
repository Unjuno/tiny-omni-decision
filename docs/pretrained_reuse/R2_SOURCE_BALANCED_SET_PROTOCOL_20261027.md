# Candidate-balanced set objective comparison

The existing set objective averages BCE across all candidates on visible
records. In the pinned 1,901 visible-source training records, the mean positive
candidate fraction is 0.128; there are 3,250 positive and 25,530 negative
candidate instances. This imbalance is an observation, not proof of the error
cause. Test exactly one alternative objective: retain the same none-visible
BCE, and for visible records average the positive-candidate BCE and the
negative-candidate BCE with equal 0.5 weights. If a class is absent, use the
mean of the present class. No changes to model architecture, features, folds,
seed 17, modes, threshold rule, AdamW, batch size, seven epochs or schedule.

Repeat the same 3 video-group folds, three audio/video/joint modes and pinned
train data as the existing source set CV run. Save nine checkpoints, per-record
predictions and hashes in a fresh output. Compare joint and unimodal exact-set
accuracy, per-stratum outcomes, candidate and none NLL/Brier/ECE, and paired
joint-versus-old-joint results on matching IDs using a 320-video bootstrap
(10,000 draws, seed 17). Fixed 0.5 decisions, no tuning or model selection.
Balanced BCE changes the implied probabilities; do not assume they are calibrated.
The evaluation is nonblind reused train-source development with oracle tracks
and possible contributor overlap, not independent transfer or product quality.
No sealed audit, new data acquisition, early stopping or model promotion. Keep
all prior artifacts and fail closed on group, input, output or disk errors.
