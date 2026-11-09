# Speech Commands fresh-speaker M1 readout protocol — 2026-11-10

## Question

Can one fixed training run of a small readout on frozen Whisper-Tiny features
improve the narrow ten-keyword audio Decision result on a new, speaker-disjoint
development cohort, compared with the frozen calibrated readout that scored
93.8% on the already-observed 1,000-clip cohort?

This experiment only tests English keyword classification. It does not test
general speech understanding or multimodal fusion.

## Frozen source and exclusions

- Use only `google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da`,
  v0.02 official TRAIN shards and the fixed class order
  `yes,no,up,down,left,right,on,off,stop,go`.
- Exclude every speaker group, source row, and source path recorded in the
  historical 1,920-row Speech Commands cohort.
- Also exclude every speaker group, source row, and source path from the
  observed fresh-speaker evaluation cohort (manifest SHA-256
  `06baea7a101ce60849b5a6dc81d4413c531c0871091fa914d169a53e5f30f23e`).
- Do not read official validation/test shards or any sealed audit data.
- Do not use the observed 562-speaker cohort for training, calibration, or
  selection. It is already observed and remains frozen.

## Deterministic group and row rule

1. Scan only TRAIN metadata columns `label`, `speaker_id`, and `file`, with
   source row indices. Verify the Parquet ClassLabel order against the frozen
   class list. Do not read audio during preflight.
2. Hash speaker IDs and remove the complete historical and observed groups.
   Rank remaining groups by
   `SHA256("r2-speechcmd-m1-seed17|" + speaker_group_sha256)`.
3. Reserve 200 development groups with a deterministic rare-class-coverage
   greedy rule: at each step choose the group covering the largest number of
   labels that still have fewer than 100 distinct development speakers; break
   ties by the hash rank above. Fail if the 200 groups cannot support 100
   distinct speakers for every label. The remainder are the training pool; the
   existing training procedure makes a speaker-held-out 20% calibration
   subset within that pool using seed 17.
4. Select one row from each of the 200 development groups, always assigning
   the currently least-filled available class (SHA-256 tie order). Then fill
   each class to exactly 100 rows by
   `SHA256("17|dev-fill|class|source_file|source_row")`, at most one row per
   speaker per class. The final development cohort must contain all 200 groups.
5. From the training pool, select exactly 150 rows per class, at most five
   selected rows total per speaker. Rank each class queue by
   `SHA256("17|train|class|source_file|source_row")`; walk the queues in fixed
   class order, taking one eligible row per class per pass and skipping a
   speaker after its fifth selected row. This prevents earlier classes from
   consuming the entire speaker budget before later classes are sampled.
6. Fail closed before media retrieval if fewer than 200 development speakers
   remain; any class lacks 100 distinct development speakers or 150 eligible
   training rows under the speaker cap; the training-pool speaker split lacks
   100 fit speakers or 30 calibration speakers; or the fit/calibration split
   lacks 20 examples in any class. Do not reduce quotas or weaken a gate.

## Metadata-only protocol correction

Before any media retrieval, the initial implementation exposed two sampling
problems. Its training selector incorrectly required 150 distinct speakers
within every class, although the frozen protocol allows up to five selected
rows total per speaker. After that code defect was corrected, selecting the
first 200 groups by hash left only 93 distinct development speakers with the
`left` label. No audio, model, inference result, or training data was accessed.

The protocol therefore uses the deterministic coverage rule above and a
round-robin walk of per-class hash-ranked training queues. These changes
preserve the 200-group, 100-per-class, 150-per-class, and five-rows-per-speaker
quotas; they do not use model metrics or audio content. The original failed
metadata-only attempts remain in the external artifact directory and are not
overwritten.

## Single fixed run

- Frozen encoder: `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af`,
  SHA-256 `d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8`.
- Train only the existing 3,850-parameter linear readout.
- Seed 17; existing eight-epoch AdamW schedule (`lr=1e-3`, weight decay
  `1e-4`, batch 64); use the final epoch, no validation checkpoint selection.
- Fit one scalar temperature only on the internally held-out, speaker-disjoint
  calibration subset of the training pool.
- Evaluate once on the 200-speaker development set. Compare against the frozen
  prior checkpoint and its preexisting temperature `0.42702477413437295` on
  those exact same IDs; also report uniform and training-majority baselines.
- Report Accuracy, NLL, Brier, ECE-15, mean confidence, per-class recall,
  speaker-bootstrap intervals, save/reload parity, and all hashes.
- No further seed, epoch, class-weight, threshold, or temperature search on the
  development cohort. Do not promote a candidate based on this one result.

## Stop and integrity rules

Before acquiring audio, save the metadata-only cohort plan and prove zero
speaker/source-row/source-path overlap across historical, observed, training,
calibration, and development groups. After media materialization, hash-check
all WAVs and reject exact cross-split audio-content duplicates. If any support
or overlap gate fails, stop without loading the model or fitting a readout.

Use a new artifact directory under
`C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-fresh-m1-20261110-...`.
Do not modify previous model, media, feature-cache, or evaluation artifacts.
Local compute only; no paid cloud. This is a task-specific M1 experiment and
does not close R2-M, the four-modality quality gate, or the project objective.
