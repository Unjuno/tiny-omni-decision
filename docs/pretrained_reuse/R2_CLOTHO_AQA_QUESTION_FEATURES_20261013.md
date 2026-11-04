# Question features and unresolved filename provenance

Executed frozen pretrained MiniLM on all3,422 questions in the fixed development
splits, without fitting or scoring. Model revision
`4ca70771034acceecb2e72475f72050fcdde4ddc`, actual17,389,824 parameters;
weightSHA `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
Local-only model/tokenizer load, CPU FP32 eval, gradients disabled. Final hidden
states use attention-mask mean pooling. No truncation: maximum token counts
train19, calibration17, validation24. Finite384-dimensional features saved for
each question in original split order. Extraction1.939194s excluding model load.

Artifact directory:
`C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-question-features-01/`.
Exact executed script is `clotho_aqa_question_features.py` in its parent.
Adjacent report contains split input/feature hashes and parameter count. This
enables a question-only control before audio+question fitting. No answer metrics,
readout training, hyperparameter selection or official test access occurred.
Audio+question claims must exceed question-only and audio-only controls on the
same development IDs; feature extraction success is not decision quality.

## Additional provenance investigation

Read official AquaNet source at
`a3e02c6e64efdec5c9e264e3a3537755348ed9f5` (extract_features, data_generator,
split_dataset, README) and Clotho dataset code at
`8a1981da4722a60cb3ed6424e3613edbaa124d82` (dataset, file_io, README).
No correction for `souffle_me_tallique.wav` was found in those files. AquaNet
extracts features from directory filenames directly. This limited inspection
does not establish that no correction exists anywhere. The archive alias is
still unverified; no fuzzy substitution or sample removal has occurred.
Fetched source files, revision trees and SHA ledger are saved under
`C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-code-audit-01/`.
The source was inspected only, not installed or executed.

Official repositories: https://github.com/partha2409/AquaNet and
https://github.com/audio-captioning/clotho-dataset .
