# Original GRID provisional development split

Freeze before generating a split. Read only the pinned original alignment ZIP
and the existing Lombard **train, calibration and observed development-
validation manifest identities**. Do not inspect predictions or any sealed
audit. Exclude every original GRID code occurring as an actual code in those
three Lombard manifests. Exclude original speaker `s1` (reserved for technical
smoke) and `s21` (publisher says video is missing). The remaining 32 source
speaker labels are assigned 24 train and 8 development-validation by sorting
SHA-256 of the literal namespace `grid-development-v1|` plus the speaker label;
no selection by target, media, model result or identity resemblance. Parse all
original alignment words and reject malformed entries or disagreement with
the six-character code; fail closed on duplicate code or missing speakers.
Persist selected IDs, labels, source hashes and sorted speaker assignment to a
new external output. Verify zero code and source-speaker overlap between train
and validation, and zero code overlap against all three Lombard manifests.

This is a provisional **development** split, not a sealed audit or evidence
of independent people across corpora. The original/derivative speaker label
mapping differs, and physical speaker overlap with Lombard is unverified.
Publisher statements and license apply to the Zenodo source. Audio/video
availability, byte hashes, alignment and successful decoding must be verified
before any model run. Do not count unverified alignments as media examples,
interpret speaker numbers across sources as person identity, or promote a model
based on this split. Actual video archive acquisition begins only after the
single-speaker smoke establishes the required format and runtime.
