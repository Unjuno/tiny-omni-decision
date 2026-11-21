# EVE Earth Observation MCQA TRAIN-only grouped CV protocol

## Scope

Measure the existing frozen MiniLM-L3 plus 196,865-parameter shared candidate
readout on a previously unused, expert-reviewed, text-only Earth Observation
multiple-choice task. This is an additional narrow text task, not evidence of
image understanding, multimodal fusion, or product-wide quality. The source
publishes only a TRAIN split; no development/evaluation split or sealed audit
will be opened. All reported quality metrics are out-of-fold estimates from
grouped cross-validation over that TRAIN split.

## Frozen source and integrity gates

- Dataset: `eve-esa/mcqa-single-answer` at
  `f99c5779ca51e6d3407eeebe8297d5081af8a66b`.
- Data card declares CC BY 4.0. The EVE paper says candidate examples were
  created by humans and LLMs and reviewed/refined by 25 EO/Earth Sciences
  experts. This source task is limited to the published 1,261-row MCQA shard.
- Read only `data/train-00000-of-00001.parquet`, SHA-256
  `9f711235b688491f27a0b484f6fc31bd3fba1ede261ffc2d500a73f896833509`,
  1,082,818 bytes. The only other source file is the README/card; no separate
  validation/test payload exists in the pinned repository revision.
- A/D/B/C answer labels map to the corresponding supplied option positions;
  never feed the answer letter or its position as a model feature.
- The source does not expose question-source document or annotator identity.
  Group repeated normalized question text and repeated normalized question +
  unordered choice content transitively. This prevents exact-content overlap;
  it cannot prove document/author independence or remove semantic near-dupes.
- Preflight found four structurally invalid rows (zero-based source rows 20,
  578, 821, 893): each has a null/empty option at the position named by its
  answer label. Exclude these before feature extraction or fitting because the
  target option does not exist in the supplied candidate list. This is a
  source-integrity rule, not a performance-based filter. Preserve all four
  fingerprints in the external preflight receipt; evaluate the remaining
  1,257 rows and report the exclusion count.
- Save ordered IDs, group IDs, outer/inner assignment, source/config/model
  hashes before fitting. Keep raw data, frozen feature tensors, checkpoints,
  histories, and per-example predictions outside Git.

## Frozen preflight receipt

- All 1,261 source rows were schema-audited; 1,257 are structurally usable.
- The usable rows yield 1,257 content groups. Exact normalized question and
  exact normalized question-plus-unordered-choice duplicates both number zero.
- Label-blind five-fold assignment sizes: 252, 252, 251, 251, 251 rows/groups.
- Usable answer positions (zero-based): 0:351, 1:285, 2:316, 3:305.
- Usable source ID order SHA-256:
  `d416f65078018c3c41ef1533ac286c08369453f758a95decc45680dd6ccb708e`.
- Group/fold assignment SHA-256:
  `19be1051f06c96777e397de3e377e02ee09bbe8ab36dae3adc80cd672ad996b9`.
- The four malformed-row fingerprints and reasons are saved in the external
  `preflight-only.json`; its SHA-256 is recorded in the experiment results.
  Frozen preflight receipt SHA-256:
  `0b8e99e3d743a082daed794a016a2adbc13070e9fd857dbfb6f10d1a98056a6f`.

## Fixed experiment

- Seed 17, five outer folds, deterministic label-blind SHA-256 ordering and
  balanced round-robin assignment by connected group.
- Within each outer-training partition, select one 20% inner stop fold by a
  second deterministic SHA-256 ordering of groups. Fit only on the remaining
  outer-training groups. Select the earliest minimum inner-stop NLL checkpoint
  from epochs 1–8; score the outer fold once with that checkpoint.
- Frozen model: `sentence-transformers/paraphrase-MiniLM-L3-v2` at
  `4ca70771034acceecb2e72475f72050fcdde4ddc`; use the existing local snapshot
  and verify weights hash `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- Inputs: question and each supplied choice only. Preserve source choice order;
  use the existing question/option feature construction and shared readout.
- Readout: 128-hidden GELU MLP; AdamW, learning rate 0.001, weight decay
  0.0001, batch size 64 questions, max sequence length 256, 8 epochs. No
  augmentation, architecture sweep, seed search, or external retrieval.
- Do not inspect source fields beyond Question, Choices, and Answer needed to
  normalize the task. No public test, sealed audit, or unmanifested split.

## Reporting and decision rule

Save per-fold and pooled out-of-fold Accuracy, NLL, multiclass Brier, ECE-15,
answer-option count, and the uniform and train-position-prior references.
Report 10,000-replicate seed-17 bootstrap intervals resampling connected
content groups. Preserve every OOF prediction and selected epoch outside Git.
Do not tune on pooled OOF results. This single fixed evaluation either provides
task-specific transfer evidence or retires this source; it cannot close the
text, image, audio, video, fusion, quantization, package-size, or release gates.
