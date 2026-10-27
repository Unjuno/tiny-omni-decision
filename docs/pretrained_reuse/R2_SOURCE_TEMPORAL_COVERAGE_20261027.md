# Perception source-candidate cached temporal coverage

The pinned existing feature plan was inspected for its **training partition**
only: 2,593 event records, 320 video groups, 38,111 visual candidates and
5,852 two-second audio windows. This reads plan metadata, not the media or
feature values. The plan file contains the other partition's metadata as well,
but this diagnostic selects and summarizes only `train.records`; no development
labels/features, model or sealed audit was opened.

- **1,852/2,593 records (71.42%) have just one audio feature window.**
  The remainder have 2–18 windows.
- **22,602/38,111 candidates (59.30%) have only one indexed video frame.**
  The rest have two (5,837), three (2,581) or four (7,091).
- **1,667 records** have at least one single-frame candidate.
- **2,591 records** contain at least one padded audio window; 2,591 of the
  5,852 windows have some padding, and 1,722 windows are more than half padding.
  This does not mean that every record or entire recording is mostly silence.

The existing readout mean-pools the audio windows and represents each video
candidate by its frame mean and last-minus-first difference. A one-frame
candidate has no observed visual difference. This metadata makes it clear that
sequence lengths and padding masks vary substantially; a temporal model must
not mistake padding, repeated frames or synthetic interpolation for observed
motion. An LSTM can still be compared, but long-range sequence learning is not
identifiable for a single-window/single-frame case. The plan does not establish
physical audio/video clock alignment, source-label truth or independent groups.
Inspect actual timestamps and masks before claiming synchronized fusion.

The output at
`C:/CodexArtifacts/pretrained-reuse-r2/perception-source-temporal-coverage-20261027-attempt-02/result.json`
has SHA-256 `e15c92ccc7287b0372aa265cd049adf6c1e525b1d0918b29750ac7df316b55bf`.
It records the complete length distributions, plan and script hashes. The first
attempt recorded lengths only; the second added the amount of padding under the
same plan and checked the two-second window invariant. Both outputs are retained.
Two targeted tests and Ruff passed. This is a coverage diagnostic, not model
training, a quality score, a bottleneck proof or evidence for a parameter change.
