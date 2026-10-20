# CLEVRER synthetic audio-video task feasibility preflight — 2026-10-20

## Decision

CLEVRER is a possible rights-clean base for a controlled audio-video binding
diagnostic because its official training corpus is identified as CC0-1.0 in
`manifests/candidates/clevrer.yaml`. This note does **not** admit a new corpus,
claim that CLEVRER contains audio, establish audio-video necessity, or authorize
a model fit. CLEVRER source videos are visual simulations; any audio would have
to be generated and explicitly described as synthetic.

The next safe step, if this direction is selected, is a complete exposure and
split audit over all existing project metadata and the official train question
manifest. Do not treat this preflight's upper bound as an untouched-sample
count.

## Read-only exposure scan

The scan examined exactly named `train.jsonl` and `validation.jsonl` files
under `C:\CodexArtifacts\pretrained-reuse-r2`, excluding any directory path
containing `test`, `sealed`, or `audit`. It read only rows tagged with
`MIT-IBM/CLEVRER@...` or a `clevrer:` scene group and collected split plus
scene index. It did not inspect media, raw question payloads, predictions,
official test data, or sealed-audit data. No file was modified.

| Measure | Observed |
|---|---:|
| Metadata files scanned | 84 |
| Matching metadata rows | 18,433 |
| Distinct CLEVRER train scenes represented | 2,423 |
| Distinct CLEVRER validation scenes represented | 1,763 |
| Official train scene count in the pinned candidate manifest | 10,000 |
| Arithmetic remainder before other repositories/manifests are reconciled | 7,577 |

The train scene-ID set digest for the scan is
`dd848c08147507617aa5b3e449959318b78e043e79de14be6ad4cf4920b8ff28`.
This is a digest of the comma-joined, ascending scene indices, not a dataset
manifest hash. The remainder is only an **upper bound**: the scan did not cover
every worktree, repository manifest, original source file, or all possible
artifact layouts. It is not evidence that 7,577 scenes are untouched.

## Experiment implications

- A synthetic audio overlay could provide controlled timing cues, but the task
  would measure synthetic cross-modal alignment, not natural audiovisual
  understanding.
- The task must use scene groups disjoint across train and development, and
  exclude every scene already present in project corpora and run artifacts.
- Questions, options, audio-generation seed, event-to-sound mapping, and answer
  derivation must be frozen before feature extraction or training.
- Both-input necessity must be checked with matched video-only, audio-only,
  audio/video swap, timing-offset, missing-stream, and stale-stream controls.
- A generated sound mapping that can reveal the answer from audio alone is an
  invalid joint task. A question template or answer construction that reveals
  it from text alone is likewise invalid.
- Only after those preflights pass should frozen VideoPrism and an approved
  audio encoder be considered for a small readout. It would remain a synthetic
  diagnostic and would not close the natural-AV R2 gate.

## Reproduction boundary

The scan used Python 3.11, standard-library `os.walk`, JSONL parsing, and SHA-256
over the sorted train scene IDs. It selected filenames `train.jsonl` and
`validation.jsonl` only, skipped virtual-environment/hidden/temporary
directories, and pruned paths containing `test`, `sealed`, or `audit`. No model
was loaded, no GPU was used, no data split was created, and no experiment
started. Existing corpora, checkpoints, feature caches, and audit sets remain
unchanged.
