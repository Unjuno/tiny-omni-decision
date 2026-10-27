# Blinded train-source audio/video review sheet

A local review sheet was generated from the 16 frozen train-source records. It
checks each original MP4 against the frozen sample hash, orders records using a
second deterministic hash and omits original label categories and model
predictions. The sheet lets a reviewer record whether the sound was audible,
whether a related object or contact point was visible, whether multiple
objects were involved, and whether the judgment is uncertain. Answers remain
in the browser's local storage until the reviewer exports a JSON file. They do
not change the corpus or model.

The generated JavaScript passed a syntax parse; unit tests check deterministic
order, omission of category labels and failure on changed media. **Actual
browser rendering, video playback and audio have not been validated.** The
in-app browser explicitly rejected `file:` URLs under its security policy, so
no alternative route was attempted. No human judgments were obtained; the
annotation-semantic gate remains open. The sample is stratified and nonblind
at the dataset-design level, not a representative or final audit.

Local sheet:
`C:/CodexArtifacts/pretrained-reuse-r2/perception-source-semantics-review-sample-20261027-attempt-01/review.html`,
SHA-256 `707c8e00bc7a27d39cfbd0b33f0001de8f1a935157f79698abf221d607d8efb5`.
It contains local references to the existing MP4 files and no external service;
no raw media or responses were added to Git. No model was trained or evaluated.
