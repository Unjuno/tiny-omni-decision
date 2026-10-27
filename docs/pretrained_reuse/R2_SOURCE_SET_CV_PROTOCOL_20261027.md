# Source set-valued readout: fixed train-source comparison

Compare fresh `CandidateModel` readouts with the new visibility/membership
loss on the same frozen train-only source features and video-group folds as
the existing unimodal CV run: 2,593 records, 320 videos, seed 17, 3 folds,
7 fixed epochs, AdamW 0.001, weight decay 0.0001, batch 32, clip norm 5,
identical per-fold initialization and order for audio-only, video-only and
joint. Frozen encoders and 344,834 readout parameters are unchanged. Run all
nine arms once; no early stop, seed sweep or threshold tuning.

Use the fixed 0.5 conditional candidate and none threshold with a nonempty
fallback selecting all candidates within 1e-6 of the maximum logit. Save
per-record candidate IDs, labels, conditional probabilities,
none probabilities and predicted sets, each checkpoint and all hashes.
Report exact-set accuracy, none-visible detection, per-candidate marginal
binary NLL/Brier/ECE and macro per-record scores; avoid treating correlated
candidate rows as independent samples. Candidate marginal is
`(1 - P(none)) * P(candidate | visible)`; this factorization does not define
a normalized joint distribution over nonempty sets. Evaluate held-out folds
once per fixed arm. Comparisons to earlier softmax results are exploratory,
with differing training objectives and decision rules. Video group bootstrap
is descriptive; contributors and scenes can overlap between folds, and the
source data and earlier evaluations have already been observed. No product
selection or sealed audit is performed. Fail closed on hashes, disk below 1 GiB,
nonfinite loss/gradients, split overlap, mismatched IDs or existing output.

The audio-only model receives identical zero visual input for each candidate;
its conditional scores are theoretically tied. The decoder normally selects
all candidates when visibility is chosen, including on fallback; floating
point differences near the threshold can still affect selections. Report this
limitation and do not interpret it as independent visual source identification.
