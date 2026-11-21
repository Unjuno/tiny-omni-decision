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

## Attempt 02: fixed terminal result

The local-C attempt completed all 2,048 updates and the fixed terminal
checkpoint reloaded with exact logits (`reload_max_abs=0`). No checkpoint was
selected using the holdout. The same preflight split hashes as attempt 01 were
reproduced.

| Arm | Macro accuracy | Macro NLL | Macro Brier | Macro ECE | Overall accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original state | 13.04% | 2.3433 | 0.8996 | 0.0975 | 12.98% |
| Text state zeroed | 11.19% | 2.3870 | 0.9070 | 0.0617 | 11.15% |
| Same-task state swap, donor target | 14.45% | 2.3270 | 0.8973 | 0.1211 | 14.42% |

The candidate set has 40 choices, so uniform choice accuracy is 2.5%; the
13.04% result is above that reference, but remains poor decision quality and
is not product evidence. The minimum task Accuracy was 4.55%. Original-state
Accuracy by attribute (macro over the 12 corresponding question/format strata)
was color 12.79%, count 12.25%, shape 11.94%, and texture 15.18%.

| Paired group intervention (control minus original) | Difference | 95% group-bootstrap interval |
| --- | ---: | ---: |
| Zero-state Accuracy | −1.84 pp | [−3.03, −0.69] pp |
| Zero-state NLL | +0.0437 | [+0.0265, +0.0621] |
| Zero-state Brier | +0.00744 | [+0.00166, +0.01382] |
| Swapped-state Accuracy, donor target | +1.42 pp | [−1.85, +4.88] pp |
| Swapped-state NLL, donor target | −0.0163 | [−0.0653, +0.0340] |
| Swapped-state Brier, donor target | −0.00233 | [−0.01347, +0.00916] |

Removing the state modestly worsened ranking/calibration losses and reduced
Accuracy; this supports a small state contribution. Replacing it with another
same-task group's state and mapping the answer by option ID did not produce a
reliable difference: all paired group-bootstrap intervals include zero. Thus
the run does **not** demonstrate robust state-to-candidate binding. The
counterfactual arm's point metrics are not evidence of improvement because the
target changes with the donor state and its paired differences are uncertain.

The 128-update rolling training CE decreased from 2.9253 to 2.3081; rolling
training Accuracy rose from 7.42% to 15.63%. These are changing training
windows, not a fixed generalization-gap estimate. No learning-curve checkpoint
was selected on the held groups.

### Run and artifact record

- Execution: `PASS_TRAIN_ONLY_GROUP_HOLDOUT_COMPLETE`; fresh seed-17
  initialization; 2,048 updates; 8,192 draws, 7,152 unique fit rows and 1,040
  repeated draws; fixed terminal checkpoint at step 2,048.
- Start/end: `2026-11-21T05:50:19.736851Z` to
  `2026-11-21T05:51:26.846073Z`; 48.57 s training, 18.52 s evaluation, 67.12 s
  total. Local RTX 3080 Laptop; peak allocated/reserved CUDA memory 106.3/110.0
  MiB; cloud cost $0.
- Trainable readout: 889,345 parameters. Counted prequantization union:
  196,254,874 parameters, within the provisional 200M parameter ceiling. The
  final checkpoint is 3,566,456 bytes; this is not a packaged or quantized
  product model.
- Configuration SHA-256:
  `dfc93f6c1c3650e7fb7833b4e7da347e8c06c4a077466772c00bfdeff1dc0fdb`.
- Source commit: `f5a80fcfac325983bdedb005a56df0a1a3f98b4f`.
- Preflight SHA-256:
  `06a6bf44a0be2a156a09b47f6661c22f9bab00878a94a3f7da7e03302f5aded5`;
  split SHA-256:
  `9744c76c610dc21f7cd5fc6031357ec6a007673c81cc0c8cf1ee3a39b6183b32`.
- C: staging-receipt SHA-256:
  `c409025b3df07f2f012bccc60ae50a4296f149fbc86125fe80f32526ed0f5109`;
  856,979,629 bytes copied with all seven source/destination hashes matching.
- Run result SHA-256:
  `bb7ac685f53cfe8762561ee2a05fe1f7b3d3e3acd973cc157289c440d4e6daf0`;
  checkpoint SHA-256:
  `23d2423838ffddf12e2add846deaf7c412f51ab9c381e56e127d3eb3095eb902`.
- Holdout metric SHA-256:
  `78c9e9ed167cee5d144cebc59afd78f55737441458678d0467a60641d6d34d2b`;
  prediction-array SHA-256:
  `0716ff1ce1ede1628128dbdf30b339657e0807861232861a2b2392a49b14ce7d`.
- Training-history SHA-256:
  `28e7f450a2ec2b1684e2ce4d29183163ce6f49b1e3ebde2e70eb34d7bb124a0f`;
  sampling-order SHA-256:
  `bc2d58e1e2518bdc7779f09ae988b0e8bc0845e3322927e946a2cbcfb912e91e`.
- Run directory:
  `C:\CodexArtifacts\pretrained-reuse-r2-local\native-text-state-group-holdout\runs\attempt-02`.

This is still only a group-heldout slice of the already frozen TRAIN corpus,
with familiar prompt templates and candidate vocabulary. The official
validation and sealed audit were not opened. The limited accuracy and uncertain
counterfactual effect do not justify expanding this synthetic task or adding
capacity; the next step should use a distinct, still-unused TRAIN decision
source and keep this holdout retired from model selection.
