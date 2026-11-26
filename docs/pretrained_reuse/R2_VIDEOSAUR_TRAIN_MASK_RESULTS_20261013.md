# VideoSAUR natural-video mask diagnostic results — 2026-10-13

Completed the [fixed protocol](R2_VIDEOSAUR_TRAIN_MASK_PROTOCOL_20261013.md).
8 train videos,4,979 consecutive frames,164 later annotated points. No examples
removed. Frozen model only; no validation, sealed audit or additional learning.

## Measured comparison

| Pooled later-point IoU | CLEVRER StoSAVi | YT-VIS VideoSAUR |
| --- | ---: | ---: |
| Fixed frame0-selected slot |.061294|.138612|
| Per-frame oracle rematching |.174399|.341503|
| Static initial box |.260975|.260975|

Same video IDs, point order, frame IDs and rasterized static geometry verified
exactly. Equal-video mean differences: fixed+.076604, oracle+.150089. Fixed-slot
IoU improves in7/8videos; oracle in8/8. These ordered eight training examples are
not a population estimate or an independent quality evaluation. No significance
claim. Models, input resolution, pretraining and device differ: not a one-factor
pretraining experiment. The comparator's privileged initial-box geometry means
static-box IoU is not a fair segmentation tracker benchmark.

| Video | Later points | Fixed IoU | Oracle IoU |
| --- | ---: | ---: | ---: |
|10103|15|.237875|.237875|
|10186|21|.100093|.346084|
|10212|35|.166213|.525587|
|10387|18|.010856|.064011|
|1087|19|.225322|.330685|
|10886|20|.097028|.386401|
|10894|18|.177258|.421514|
|10903|18|.090947|.223589|

The gap between fixed and oracle slots indicates a persistent identity/assignment
problem, while imperfect oracle localization also remains. This is not proof of
one unique cause: object splitting/merging, mask geometry and domain transfer may
all contribute. It is not Decision Accuracy and does not satisfy95–98% targets.

## Integrity and resources

- Every saved mask reaggregated exactly with existing hashed annotation and metric
  helper; no mask/annotation changes. Saved input hashes checked before inference.
- Frozen state exactly unchanged after the run. All outputs finite.
- Per-video initial slots sampled once with GPU seed17; no later reseeding.
- Chunked recurrence separately matched upstream whole-sequence processor exactly
  on synthetic9-frame CPU features; result at videosaur-recurrence-check-01/.
- GPU FP32 on RTX3080Laptop16GB;468.616seconds sequential per-video work, including
  hash/decode/preprocessing, excluding model startup. About0.130GPU-hours of
  process wall time, not a measured energy estimate. Cloud cost0.
- Peak allocated705,276,928bytes; reserved897,581,056bytes. One resident model.
- Param count93,530,201, trainable0. Runtime/model provenance limitations and
  isolated package versions remain those of the R1 runtime audit.

Root: `C:/CodexArtifacts/pretrained-reuse-r1/videosaur-train-masks-01/`.
Contains protocol, progress, result, eight mask caches, replay and comparison.
Protocol SHA22fe900e41f7cec674b8703cf0ca75f0e6e316ab034d68e858d06bb61b98f80f.
Result SHA4e6b245ab64fb803e16c8cc828926764badb094bdc183a75a20d0a1496b73531.
Harness SHA333dbbe8adeb837e253d9cd6b8571ab62a2af95e034162d66b7fee0c0c038036.
Parent repo commitcdf5c98; source/harness/config hashes preserved in protocol.
Small comparison JSON is committed; raw masks/weights stay outside Git.

Reproduce metrics with `python -m scripts.replay_slot_mask_diagnostic` using the
run root, original train annotation archive and a fresh output path. Compare the
old/new result.json files with `python -m scripts.compare_slot_mask_diagnostics
--reference <old-result> --candidate <new-result> --output <fresh-output>`.
Comparison rejects video/frame/geometry mismatch and nonfinite IoU; self-comparison
of the actual old results gives all-zero differences. Tests673passed,1skipped;
Ruff passed. Existing weights, corpora and runs were not overwritten.

## Next evidence gate

Do not launch a full Decision fit yet. A bounded annotation-free mask-association
control on saved masks is cheaper and directly tests whether fixed slot IDs are
losing identity that mask geometry could preserve. Only frame0 initialization may
use the marked box; subsequent assignments must not use the correct boxes. Any
such control is still an oracle-initialized diagnostic, and sparse saved frames
limit its interpretation. Do not tune matching parameters/seeds on these results.
Positive evidence would justify a separately specified Decision representation
probe; it would not itself satisfy the full Omni product goal. Redistribution
rights and exact original non-shape training settings remain unresolved.
