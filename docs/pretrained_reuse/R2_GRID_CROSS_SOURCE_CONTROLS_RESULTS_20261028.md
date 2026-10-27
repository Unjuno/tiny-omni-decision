# Unmatched cross-source interface controls: result

The [fixed controls](R2_GRID_CROSS_SOURCE_CONTROLS_PROTOCOL_20261028.md)
completed using the same real Lombard training audio and four GRID smoke
videos as the earlier run. The original JSON was reproduced exactly.

- Reordering videos together with their opaque IDs preserved the complete
  probability mapping exactly.
- Removing audio or removing video produced the defined uniform distribution
  of 0.25 for each candidate.
- Swapping two video contents under unchanged IDs changed the probability
  vector by L1 distance `0.0057637112` for this one case.

The last observation proves only that this fixed output changes for this
particular unmatched input; it does not establish whether any answer is
correct, whether the change is meaningful, or how often content is used.
Uniform missing-input behavior is an explicit runtime policy, not a trained
unimodal result. Video logits were reused from the separately executed GRID
video/head runs; the audio encoder was run on CPU. There was no training,
target, model selection, held-out scoring or sealed audit.

Output:
`C:/CodexArtifacts/pretrained-reuse-r2/grid-cross-source-controls-20261028-attempt-01/summary.json`,
SHA-256 `41199ea205323a2f117282d561adf8b675a900dcf2e106f35936aab30ebee0fe`.
It records all outputs and pinned inputs. Four focused existing tests and Ruff
passed. Matched GRID audio/video, actual quality and independent-person checks
remain open.
