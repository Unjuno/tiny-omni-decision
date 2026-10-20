# Text-free CLEVRER scene-coverage extension protocol

**Status: matched 512/1,024-scene runs completed.** This is a narrow
frozen-feature Path A development study. It does not evaluate audio, image,
multimodal fusion, open-ended video reasoning, or product quality.

## Question and decision rule

At the same number of optimizer updates and example presentations, does
doubling unique CLEVRER training coverage from 512 to 1,024 scenes improve a
small text-free typed-video Decision readout on temporal and static descriptive
questions? The 1,024-scene arm is the candidate; the 512-scene arm is its
deterministically selected nested control. Both use the same 256-scene
development validation in the same order. The fixed selector is minimum macro
validation NLL over the question-type/taxonomy groups represented in the
validation set, with the earlier epoch winning ties.

This development validation selects checkpoints and therefore is not blind
evidence. Its repeated question templates are reported; scene is the independent
cluster. No sealed audit or official test data are opened.

## Data, provenance, and grouping

- Source: `MIT-IBM/CLEVRER` at revision
  `98b842082ba4f7c18b6b9e3f39145871782a65ef`, admitted as CC0-1.0 for training
  by `manifests/candidates/clevrer.yaml`.
- Main sample: 1,024 official-train scenes / 2,048 rows and 256 disjoint
  official-validation scenes / 512 rows. Each scene contributes one temporal
  and one static descriptive question.
- Nested control: 512 of those 1,024 training scenes / 1,024 rows and the same
  256 validation scenes / 512 rows, selected by the frozen SHA-256 scene order
  with seed `20261018`.
- Main sample JSONL SHA-256: train
  `61e4ca966dc431e3cce6832860ed54cc861945ab1e1876614367304eb0adadd3`,
  validation
  `2ff03047e7c763cadb975058b842e88102fa96d1327e4c2fa32dfef3c60b0cb5`.
- Nested train JSONL SHA-256:
  `39ac25ea7012f40ed2c74b63e69ef4f265f035bec60b96c82bc219295dbd5f6d`.
- Media SHA-256 checks passed for all 1,280 selected assets. Train/validation
  scene, media, and sample identity overlap are zero. The main sample contains
  159 repeated normalized question/option templates across different scenes.
  CLEVRER deliberately reuses these templates; the per-scene media and labels
  differ, and the template count is disclosed rather than treated as independent
  question coverage. Template recurrence is explicitly permitted only for this
  frozen scene-grouped experiment.
- External data root:
  `C:/CodexArtifacts/pretrained-reuse-r2/textfree-video-coverage-1024train-256val-v2/`.
  Existing samples and extraction attempts remain immutable.

## Frozen representation and readouts

Use the pinned VideoPrism-B checkpoint and existing frame-position readout
contract: eight uniformly sampled endpoint-inclusive frames, spatial-token mean
per frame, 768 features per position. The VideoPrism encoder remains frozen.
The Decision input consists of framewise video features, typed program,
taxonomy, and supplied options. Natural-language question text, target, scene
ID, sample ID, and media hash are not model inputs.

Compare fresh seed-17 `mean_mlp` and `bilstm` readouts at each coverage size.
Both use AdamW (`lr=0.001`, weight decay `0.0001`), gradient clip `1.0`,
constant learning rate, batch size 64, and exactly 2,048 optimizer updates.
Both consume 131,072 example presentations per arm. The 512-scene arm runs 128
epochs (16 updates/epoch) and validates every two epochs; the 1,024-scene arm
runs 64 epochs (32 updates/epoch) and validates every epoch. This gives both
arms validation opportunities every 32 updates and keeps the selector budget
matched. There is no early stopping. The best selected checkpoint and final
checkpoint are saved separately.

The same seed, update budget, model/readout size, optimizer, selector and
validation IDs are used for both sizes. The 512-scene arm is nested within the
1,024-scene arm. The intended changed factor is unique training-scene coverage.

## Preflight and safety

The main sample fetch receipt SHA-256 is
`9f51132faa546410186728038e838ccdb609b9b7a8d6a0938a443411a896dcf4`; its media
integrity receipt SHA-256 is
`35ee8217b67e6c513b67aa62beb1da9d91644d3897642c7cb9839e042f13e42a`. The
nested sample's fetch and integrity receipts are
`002568459beab36f67d4f94f635710d0c43de407fc2e34cdf10111f0913e7a93` and
`e8159ae3b59fe912416243a9479bc48de081fb3a32434da7f9cefd20eca3c0e8`.

The first extractor launch waited for Orbitune's active GPU process, acquired
the 3,200 MiB HotUGPU reservation after yield, then exited before reading data
because a direct script invocation could not import the local `scripts` package.
It created no output and the reservation was released. The corrected attempt
uses `python -m scripts.extract_videoprism_framewise_features` with the target
repository as its working directory, through the same cooperative guard. It
does not stop or signal Orbitune. Feature extraction completed for 1,280 unique
videos (2,048 train rows, 512 validation rows), in 1,640.5 seconds on the JAX
`TFRT_CPU_0` backend; peak working set was 4,162,793,472 bytes. The compressed
`[1,280, 8, 768]` feature cache is 29,352,955 bytes, SHA-256
`504f50303e1062344b1dcba3ad251974dc2c47d634572b1628582baf2396f5f5`.
Checkpoint, source revision, manifests, scene/media separation, and cache hashes
passed preflight. The two CPU smoke runs produced finite losses and gradients;
the guarded CUDA smoke also passed for both readouts (226,865 and 227,377
parameters). One GPU reservation handoff safely aborted when Orbitune's process
reappeared; a later guarded attempt succeeded after the GPU yielded. No process
was stopped or signalled.

The first frozen-receipt preflight exposed a runner key mismatch (`validation.jsonl`
versus the recorded `validation_jsonl`). The reader was corrected to match the
fetch receipt schema and a regression test was added. No receipt or dataset was
changed. Both arms passed CPU preflight/smoke and a guarded CUDA smoke. The first
full-run launcher was safely rejected twice on reservation handoff races; the
next handoff ran both arms. A PowerShell stderr policy then aborted an earlier
zero-update attempt when PyTorch emitted its Windows allocator warning. That
partial output directory was preserved as attempt 01. A corrected launcher and
new output directory completed both arms without changing their configurations.
The detailed results, curves, hashes, overfitting evidence, and paired scene
bootstrap are in the [completed results report](R2_TEXTFREE_VIDEO_COVERAGE_1024_RESULTS_20261020.md).

The existing 256-scene/200-scene baseline preflight also passed after the
runner was corrected to import this worktree's own `src/` before any globally
installed editable package. This prevents accidentally evaluating code from a
different worktree. The correction is covered by the runner's direct CLI
preflight and regression tests.

## Metrics and limits

At selection and at the final update, preserve train/validation CE, Accuracy,
NLL, Brier, ECE, question-type/taxonomy metrics, per-example JSON option
probabilities, best step, and best-reload parity. Report train and validation
curves, unique scenes/assets/examples, repeated exposures, elapsed extraction,
fit and evaluation time, peak VRAM, package/readout parameter counts, and
paired scene bootstrap intervals (5,000 repetitions, seed 17). The bootstrap
is descriptive because the same validation selects the checkpoints.

This comparison can support or fail to support a training-coverage effect for
these two CLEVRER descriptive task families. It cannot prove temporal reasoning
superiority, a model-family ceiling, general multimodal quality, or the
95–98% product aspiration.
