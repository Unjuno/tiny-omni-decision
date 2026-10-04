# Video Teacher v2 Staged Implementation Plan

## Goal

Improve video decision quality while preserving Teacher v1 as the immutable
baseline. Follow the experiment order in
[`2026-10-04-video-teacher-v2-design.md`](../specs/2026-10-04-video-teacher-v2-design.md):
first make Candidate A (8 frames, q/v, rank 16, frozen v1 corpus) work and
verify it with CPU tests and a small processor smoke; only then proceed to
Candidate B (8 frames, decoder attention+MLP targets, rank 16). Later candidates
are conditional on measured validation and memory evidence.

Teacher v1's selected adapter, source config, frozen corpus, selection lock,
curves, hashes, metrics, and already-used sealed audit are reference-only. The
recorded source config hash is
`a68fae06267078dbbbcc8afc0ce4680cb553ed28b968e64e2bde50955c5a6409`, matching
the saved `artifacts/tiny-omni-decision-teacher-v1/selected/training-config.yaml`.
That file and the v1 corpus/artifacts must not be rewritten. The v1 train and
validation JSONL files are present locally and will be reused by reference;
no new corpus build is part of Candidates A/B.

## Architecture

The existing Decision pipeline is intentionally retained:

1. `training.py` parses a strict Pydantic config and builds processor inputs.
2. `trainer.py` loads the pinned multimodal base, freezes base parameters,
   attaches decoder LoRA, performs validation-only selection, and saves an
   experiment manifest/history.
3. `corpus.py` and `dataset.py` enforce source/media/content split gates.
4. `tests/test_training.py` and `tests/test_dataset.py` cover CPU behavior.

Implementation is staged. Candidate A introduces configurable frame count
while preserving the v1 q/v target, rank, data, validation subset, LR, seed,
and other settings. Its processor smoke must show `num_frames=8` reaches the
processor and that produced sequence length still obeys the configured limit.
Candidate B target discovery must match only real `torch.nn.Linear` modules
whose full loaded path begins `model.language_model.` and whose leaf is one of
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, or `down_proj`.
No model-wide `all-linear` PEFT selector is allowed.

For any future CLEVRER task-mix work, retain the existing video-identity split
and leakage gates. Convert each upstream multiple-choice answer to the existing
single-target binary `DecisionExample`; do not introduce a separate loss path.
That dataset work is deferred until the structural frame/adapter candidates
have been measured.

## Tech Stack

- Python, Pydantic, PyTorch, Transformers, PEFT
- Existing JSONL frozen Teacher v1 train and validation corpus
- Local RTX 3080 Laptop (16 GiB) for model/processor checks and training
- pytest and Ruff for CPU CI checks
- Candidate-specific ignored artifact directories under `artifacts/`; no
  selected v1 path reuse

## Spec

### Inventory and baseline guards

- `src/tiny_omni_decision/training.py`: `DecisionTrainingConfig`, YAML mapping,
  `processor_inputs_for_example` (currently hardcodes 4 video frames), and
  `resolve_decoder_lora_targets` (currently q/v only under the language model).
- `src/tiny_omni_decision/trainer.py`: `_forward_decision` calls the processor
  helper; `run_training` constructs `LoraConfig` and a fixed-LR AdamW optimizer;
  validation selection and experiment history are already recorded.
- `src/tiny_omni_decision/corpus.py` / `dataset.py`: current asset-level split
  and overlap checks are in force. `dataset.py` currently normalizes CLEVRER
  descriptive questions; A/B must not alter the frozen corpus.
- `configs/decision/teacher_v1*.yaml`: existing baseline and search configs.
  Preserve their realized v1 values; record the old selected-run config hash.
- `manifests/teachers/tiny-omni-decision-teacher-v1.json` and
  `artifacts/tiny-omni-decision-teacher-v1/selected/`: immutable baseline
  records and weights.
- `tests/test_training.py`: config mapping, deterministic sampling, split
  grouping, validation selection, and manifest coverage.
- `tests/test_dataset.py`: CLEVRER normalization and cross-split gates.
- `docs/TEACHER_V1.md`: historical results and baseline boundary.

### Candidate A — frame bandwidth only

1. Add strict config fields for `video_num_frames`, `lora_target_policy`,
   `lr_scheduler`, `warmup_ratio`, and `use_rslora`; defaults must preserve the
   old v1 behavior (`4`, `qv`, fixed/constant LR, zero warmup, and `false`).
2. Pass configured `video_num_frames` through `_forward_decision` into
   `processor_inputs_for_example`; retain the current post-processor sequence
   length fail-closed check.
3. Add a separate Candidate A YAML copied from v1 settings, changing only
   `video_num_frames` to 8 and naming the candidate output independently.
4. Add CPU tests using a recording fake processor that verify configured frame
   count is passed only for video and defaults preserve 4; verify Candidate A
   still resolves q/v and rank 16.
5. Run Ruff and CPU tests, then a small processor smoke on a locally available
   video. Record effective frames, tensor/sequence shapes, and the sequence
   limit result without evaluating the sealed audit.
6. If the real processor/media path is available, train Candidate A in a new
   artifact path with seed 17 and the v1 frozen data/validation protocol. Save
   best checkpoint, learning history, config/corpus hashes, validation metrics,
   unique sample/asset counts, realized mix, peak VRAM, and wall time. Start
   with a 512-step measurement and extend only if validation is improving.

Candidate A commit boundary: frame-config plumbing, Candidate A config, tests,
and the smoke record. Its artifacts must be separate from the v1 adapter.

### Candidate B — target coverage only

Proceed only after Candidate A's frame propagation test and small smoke pass.
Extend target-policy resolution to `decoder_all_linear`, limited to exact
loaded `model.language_model.*` paths and the seven design-listed leaf names.
Record the exact resolved paths and trainable parameter count. Keep rank 16,
frames 8, the same v1 corpus, validation subset, and every non-target setting
equal to Candidate A so target coverage is interpretable. Never use generic
model-wide `all-linear`.

Candidate B must have its own config, tests, commit, run directory, checkpoint,
history, hashes, and validation comparison. Do not replace Candidate A or v1
artifacts. Per user clarification, Candidate B will match Candidate A's fixed
5e-5 LR schedule so target coverage is the only changed factor. The design's
cosine + 3% warmup setting may be evaluated afterward as a separate schedule
candidate, never attributed to target coverage.

### Conditional candidates

- C (rank 32 + rsLoRA) only if B still plateaus; frames 8, target set and all
  other settings held fixed.
- D (12 frames) only if A/B/C validation shows frame benefit and measured VRAM
  permits it; increase sequence length only by the amount actual processor
  outputs require. No 32-frame experiment.
- E (video-native mix) only after choosing the best structural candidate. Keep
  temporal descriptive tasks and add explanatory/predictive/counterfactual
  with approximate 20/30/30/20 weights, no synthetic duplicates, and report
  realized counts. Group every derived question/choice by source video.
- Projector adaptation is a separate later candidate only if video remains
  bottlenecked after frame/target/rank/task coverage. Encoders and projector
  remain frozen for A-E; do not full-finetune the base.

### Required candidate evidence

For each trained candidate, save validation-only per-modality Accuracy, NLL,
Brier, ECE; macro Accuracy; minimum modality Accuracy; train/validation gap;
unique examples/assets; realized modality/source/question-type mix; peak VRAM;
wall time; best checkpoint rule and step; trainable parameters and adapter
size. Video task-type Accuracy/NLL is required for E. Compare all modalities to
the frozen v1 validation baseline and reject a video gain that materially
damages text, image, or audio. Do not load or score the sealed audit during
candidate iteration.

### Test and commit sequence

1. Commit this plan and baseline-integrity check.
2. Implement Candidate A frame plumbing/config/test, run:
   `ruff check src tests` and `pytest tests/test_training.py tests/test_dataset.py`.
3. Run and record the small Candidate A processor smoke; commit separately.
4. Only after smoke success, implement/test Candidate B target restriction and
   its separate config; run the same CPU checks; commit separately.
5. Train candidates only in their own artifact directories, inspect validation
   learning curves, and make one causal change per follow-on candidate.
6. Update `docs/VIDEO_TEACHER_V2.md` with every attempted/rejected run and keep
   `docs/TEACHER_V1.md` as historical baseline documentation.

## Global Constraints

- No edits or resume under v1 run/config/corpus/artifact paths; no new v1
  sealed-audit reads or corpus rebuilds.
- Work on `codex/video-teacher-v2`, separate from
  `codex/teacher-quality-v1`; preserve the v1 branch and PR unchanged.
- No ternary conversion, Recovery LoRA, final merge, encoder unfreezing, or
  full-base fine-tuning.
- Use only the frozen v1 train and validation data for A/B. Validation is the
  only selection signal; sealed audit is excluded.
- Preserve CLEVRER video grouping and all current source, source-asset, media,
  and normalized-content overlap gates.
- No generic all-linear target selector; all target paths must be proven under
  `model.language_model.*` from the loaded model.
- Never duplicate scarce video/question records to hit a target mix ratio.
- Retain every candidate attempt, including failed/negative results.

## Review Focus

- Verify the frozen v1 config file hash still matches the saved adapter copy
  and manifest before/after implementation.
- Confirm config defaults preserve v1 frame/target/rank/LR/scheduler behavior.
- Confirm Candidate A passes exactly 8 to the processor while all other
  candidate settings remain v1-equivalent.
- Confirm Candidate B's PEFT targets are restricted to the seven exact decoder
  leaves and never include vision, audio, or projector modules.
- Confirm oversized processor sequence output still fails closed.
- Confirm independent validation, deterministic video grouping, and audit
  isolation remain intact.
- Confirm every trained candidate writes to a new path and stores complete
  evidence without overwriting an earlier candidate.
