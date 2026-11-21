# R2 native Text state-group holdout (2026-11-21)

## Question and scope

The frozen input-use probe found that the selected common native readout reacted
strongly to image-state changes, but scarcely changed when a Text state was
replaced by another state from the same source/task. This focused experiment
tests whether more Text supervision on previously unseen state groups can teach
the existing frozen-feature readout to bind a state to the supplied candidates.
It holds the readout architecture fixed; it does not test lexical transfer,
multimodal quality, or product readiness.

Only `text:condition_relevance_v1` rows from the existing frozen TRAIN corpus
are admitted. The corpus, pretrained feature banks, earlier models, and official
validation are read-only. No sealed-audit file or label is opened. The new head
starts from a fresh seed-17 initialization; no prior fit checkpoint is reused.

## Frozen protocol

- Corpus and query features are pinned in
  `configs/pretrained_reuse/native_text_state_group_holdout_v1.json`.
- The original fixed 2048-step common sampler is reconstructed from TRAIN rows.
  Every condition-relevance group it touched is excluded from the new holdout.
- Of 130 remaining state groups, the first 65 under
  `SHA256("17:native-text-state-group-holdout-v1:" + group_id)` are held out.
  All rows for a state group follow it into the same side.
- Preflight found 7,152 fit rows and 1,040 held rows. All 48 source/task strata
  are present; held rows per task range from 20 to 23. Fit rows per task range
  from 146 to 152. Fit and held IDs, state groups, and observation identities
  are disjoint. Every counterfactual donor has a different answer ID while
  retaining the exact task, question, and candidate vocabulary.
- Repeated prompt templates and candidate vocabulary are shared by design.
  This therefore estimates generalization to new state groups under familiar
  task wording, not generalization to new wording or source conventions.
- Training is a fresh `NativeCandidateReadout` at seed 17 for exactly 2,048
  optimizer updates, four single-question microbatches per update, with the
  fixed 62-update warmup and cosine schedule. Only cached, frozen pretrained
  text state and candidate features are inputs. There is no validation-based
  stopping or checkpoint selection: the terminal checkpoint is evaluated once.
- The terminal checkpoint is scored on original held rows, zeroed text-state
  rows, and same-task counterfactual state swaps. For a swap, the correct
  answer is remapped by option ID from the donor state, so candidate order does
  not corrupt the target. Scores are paired by held state group; the report
  records 5,000 group-bootstrap draws with seed 17.

The split and input preflight must finish before training. The runner is
`scripts/train_native_text_group_holdout.py`; tests cover deterministic group
assignment, identity isolation, balanced scheduling, and donor label mapping.
The exact pinned TRAIN JSONL, observation/query metadata, text query bank, and
TRAIN text-state bank are copied without modification from the removable D:
artifact volume to a dedicated local-C input snapshot. The copy receipt records
source and destination hashes; only those TRAIN-only files are copied, with no
validation corpus or audit payload. Outputs and checkpoints go only to new
directories below `C:\CodexArtifacts\pretrained-reuse-r2-local\native-text-state-group-holdout\runs`.
The runner rejects an output path that resolves elsewhere. Outputs are never
added to Git.

## Frozen-data inventory

The input TRAIN source has 8,192 condition-relevance examples in 512 underlying
text-state groups. The previous fixed sampler had drawn examples from 382 of
those groups. The 65-group held set is selected only from the other 130 groups.
The 2,048-update new schedule draws 8,192 fit examples; every fit row is used,
with 1,040 draws beyond the 7,152 unique fit rows. This is a targeted Text-only
training budget and is not directly comparable to the earlier 2,048-update
four-modality mixed run.

## Results

The first preflight stopped before training because the config omitted the two
expected feature-bank hash fields. The fields were added and the next preflight
passed. The first fit then reached step 128, but its D:-mounted output directory
became unavailable during the disk-reserve check (`WinError 3`). It had not yet
reached the step-256 checkpoint; its history file is empty and no model or
optimizer state was saved. Exact resume is therefore impossible. That attempt
is retained as a failed partial run and will not be represented as 128-step
quality evidence.

To prevent another output-loss event, the seven pinned TRAIN-only corpus and
feature files were copied unchanged to local C:. Their 856,979,629 bytes match
the expected SHA-256 values; validation rows/features and sealed audit data were
not copied. The original D: files were left untouched. The next preflight and
fit use only this local copy and a new C: output path, with a fresh seed-17
initialization. This remains the same preregistered training schedule and split
protocol, but it is a restart, not a continuation.

**Final metrics pending.** The report will be completed after the local-C
preflight and fixed terminal run. It will include receipt/checkpoint hashes,
runtime, per-task and macro metrics, counterfactual input-use results, group
bootstrap intervals, and limitations. Official validation and sealed audit
remain out of scope.
