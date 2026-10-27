# Fixed joint CV checkpoints: audio intervention protocol

Use the three already trained joint checkpoints and their exact train-only
held-out video folds. No additional training, checkpoint selection or access to
old development, official validation, test, or sealed audit. Recompute the
original input scores and require prediction identity/probability parity against
the pinned saved predictions before accepting any results.

Evaluate each held-out record with original audio, all-zero audio, and audio
from a different held-out video in the same fold. For the mismatch control,
sort the video IDs lexically, rotate the video list by one, and use the
lexically smallest record ID of the donor video as the audio feature for every
record of the recipient video. No target labels or semantic categories guide
the pairing. This ensures a different video ID, but is not a matched-class
control or a proof of independent recording sessions. Report the mapping hash.

Save per-record paired correctness and NLL; bootstrap the original-minus-control
accuracy and NLL by whole video group (10,000 resamples, seed 17). Record visible
and none-visible counts and accuracy separately. Zero input is out of
distribution, and a different-video audio can alter multiple factors; changes
show input sensitivity, not causal multimodal necessity. Existing OOF models
share training groups across folds, this is one seed and the task uses oracle
tracks, so results are development diagnostics rather than product validation.
