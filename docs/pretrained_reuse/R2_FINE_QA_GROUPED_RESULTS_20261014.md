# Executed original QA comparison: coarse versus fine spatial inputs

## Result and scope

Both fresh readouts completed **50 epochs / 500 updates** on the same 79 videos.
The predefined minimum-selector-NLL rule chose **epoch 8 for both**. Finer inputs
did not improve this fixed readout: selected Accuracy is 25% for both; matched
epoch-50 Accuracy is 45% for coarse16 and 32.5% for fine32. Both overfit strongly.
Preserve both runs and close this resolution-only fit without extending its budget.

This is an executed Decision QA experiment, following the previous
[actual fine input admission](R2_FINE_SPATIAL_AND_ROLE_RESULTS_20261014.md), not
another proxy-support census. It does **not** establish a pretrained-model capacity
ceiling, a blind evaluation result, or the requested full Omni quality. No product
model or Teacher was promoted. The <=200M prequantization and four-modality
95–98% objectives remain unmet.

[Predefined protocol](R2_FINE_QA_GROUPED_PROTOCOL_20261014.md) ·
[Complete records and hashes](R2_FINE_QA_GROUPED_RESULTS_20261014.json) ·
[Every epoch, both arms](R2_FINE_QA_GROUPED_RESULTS_20261014.csv).

## Data, initialization and selection

- All 99 previously known official TRAIN scenes and their 198 original questions
  remain in this pilot. Hash-ranked video/media identity with seed 17 assigns
  79 scenes / 158 questions to fit and 20 scenes / 40 questions to the selector.
  Both questions and all three original options stay with the video. There is no
  target-based stratification, example filtering, seed search or test simplification.
- These scenes had been used/discussed in older experiments. The 20 scenes are
  excluded from these **fresh heads**, but are not a new blind test. Actor isolation,
  encoder pretraining overlap and generalization to new question forms are unknown.
  The original two question templates and supplied options repeat by design.
- Actual paths are filtered to train before array loading. No official validation
  arrays, labels, annotation archive or sealed audit are opened. Existing mixed-split
  metadata is read only for the train records. No old trained readout is loaded.
- Keep the same frozen 256×512 point features, 32 source times, media/cuts, text
  features and canonical target mapping. Coarse context is 32×256×387; fine context
  is 32×1024×387. Source RGB/media/time and native coordinates are checked.
- Both arms use the unchanged 232,385-parameter `SpatialTrajectoryReadout`:
  hidden 64, 16 latents, four heads, fresh seed 17 and the same sampler order.
  AdamW uses constant LR 0.001, weight decay 0.01, eight video pairs per update,
  original QA CE plus existing balanced cover auxiliary weight 1. No new loss,
  capacity, option shuffle, encoder loading or encoder optimization.
- Hash-bound weak cover labels are available for 98 of 99 original scenes. Missing
  auxiliary supervision stays masked; its original questions are retained. Bboxes,
  source object IDs, labels and events do not enter the inference forward.
- Continue all 50 epochs. Primary comparison is the matched final epoch; separately
  retain the earliest strictly lowest selector stable log-softmax NLL checkpoint.
  Accuracy and controls do not select checkpoints. Best and final are separate files.

Split SHA-256: `fc2067efa2e64f40f0399bdf5e6a1c09b6d2ff62e9838bbf3cd27f01ccfbafe8`.
Contract SHA-256: `90c8eb106f35bba1772794d4c3bed63bd71cf4aed94cb0ed560ca1e061470238`.
Training/fresh source commit: `186710555043fef1977470cc6f03a8c70e9bed34`.
Independent auditor commit: `fbb584e179f2afbb85de6ab312e9ee1b4d25e2fa`.

Each arm consumed 3,950 video presentations / 7,900 original-question presentations
over 79 unique video assets / 158 unique questions. Repeated video presentations:
3,871 per arm. Selected epoch 8 consumed **632 video / 1,264 question presentations**;
do not confuse this with the complete run. Across both arms the unique fit assets
remain 79, rather than 158. The separate smoke consumed 16 videos per arm and did
not score the selector. The JSON retains ordered source IDs, hashes and every order.

## Original question metrics

All rows use exactly the same 20 selector videos / 40 original questions. Brier
is multiclass sum-of-squares; ECE uses 15 equal-width top-confidence bins. NLL uses
the original stable target log probability without a floor. Fixed-pool train and
selector gaps are measured independently of rolling minibatch loss.

| Arm/checkpoint | Fit Accuracy | Selector Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Coarse16 selected 8 | 48.10% | 25.00% | 1.116808 | 0.679696 | 0.133529 | 0.383529 |
| Fine32 selected 8 | 48.10% | 25.00% | 1.116834 | 0.679715 | 0.133579 | 0.383579 |
| Coarse16 final 50 | 94.94% | 45.00% | 2.050141 | 0.921897 | 0.376883 | 0.749637 |
| Fine32 final 50 | 91.77% | 32.50% | 2.250094 | 1.050535 | 0.499164 | 0.805128 |

For scale, uniform three-option probabilities have analytic NLL ln(3)=1.098612
and Brier 2/3. This is an analytic reference, not a newly fitted or selected model.

| Checkpoint | Initial Accuracy/NLL | Final-condition Accuracy/NLL | Both correct | Changed-target both correct |
|---|---|---|---:|---:|
| Coarse16 selected | 25% / 1.129401 | 25% / 1.104214 | 2/20 | 0/11 |
| Fine32 selected | 25% / 1.129430 | 25% / 1.104239 | 2/20 | 0/11 |
| Coarse16 final | 30% / 2.592223 | 60% / 1.508060 | 3/20 | 0/11 |
| Fine32 final | 25% / 2.101989 | 40% / 2.398199 | 2/20 | 1/11 |

Selected predictions have the same class counts [0,34,6] and do not establish
useful conditional grounding. Input perturbations affect probabilities; this does
not justify claiming that all inputs are ignored or that every video model fails.

### Learning curves and overfitting

The linked CSV/JSON retain all 100 epoch records, including unsuccessful epochs.

| Arm | Epoch | Fit Accuracy/NLL | Selector Accuracy/NLL |
|---|---:|---|---|
| Coarse16 | 1 | 48.10% / 1.065900 | 32.50% / 1.141056 |
| Coarse16 | 8 | 48.10% / 1.057511 | 25.00% / 1.116808 |
| Coarse16 | 16 | 73.42% / 0.691571 | 25.00% / 1.297660 |
| Coarse16 | 32 | 77.22% / 0.415315 | 20.00% / 2.352165 |
| Coarse16 | 50 | 94.94% / 0.132913 | 45.00% / 2.050141 |
| Fine32 | 1 | 48.10% / 1.065899 | 32.50% / 1.141054 |
| Fine32 | 8 | 48.10% / 1.057483 | 25.00% / 1.116834 |
| Fine32 | 16 | 73.42% / 0.690702 | 25.00% / 1.298581 |
| Fine32 | 32 | 78.48% / 0.374126 | 22.50% / 2.389328 |
| Fine32 | 50 | 91.77% / 0.195721 | 32.50% / 2.250094 |

At final 50, the fit-minus-selector Accuracy gaps are 49.94 and 59.27 percentage
points; selector-minus-fit NLL gaps are 1.917229 and 2.054373. Training improves
while selector NLL/calibration deteriorate. This is overfitting under this protocol,
not evidence that more epochs or a larger head should be launched on these IDs.

### Fixed selected-checkpoint input controls

No refit or checkpoint selection uses controls. Zero clears the whole point/context
state; mismatch moves both point and context to the next selector video cyclically,
retaining the original question. Initial/reverse change context appearance only;
positions and the original point-motion history remain unchanged.

| Control | Coarse16 Accuracy/NLL | Fine32 Accuracy/NLL |
|---|---|---|
| Normal | 25.0% / 1.116808 | 25.0% / 1.116834 |
| Whole state zero | 32.5% / 1.125751 | 32.5% / 1.125765 |
| Whole state other scene | 27.5% / 1.109679 | 27.5% / 1.109706 |
| Initial context appearance broadcast | 25.0% / 1.116936 | 25.0% / 1.116962 |
| Reversed context appearance | 25.0% / 1.116808 | 25.0% / 1.116834 |

The JSON contains all control Brier/ECE/condition metrics. The candidate-reversal
guard preserves scores within predefined 1e-5/1e-6 and original answer identity
in every evaluation batch. Reversing context alone is not a full-video time-order
ablation and cannot show that the frozen motion encoder ignores time.

### Paired uncertainty

20-scene paired bootstrap, 10,000 repetitions, seed 17, percentile 95%; retain each
scene's two questions together. These intervals are development diagnostics on
known scenes and include the influence of selector checkpoint selection.

| Fine minus coarse | Accuracy difference / interval | NLL difference / interval | Brier difference / interval |
|---|---|---|---|
| Matched final 50 | -0.125 / [-0.275,0.025] | +0.199953 / [-0.228475,0.654072] | +0.128638 / [-0.035836,0.294539] |
| Selected | 0 / [0,0] | +0.000027 / [-0.000020,0.000070] | +0.000019 / [-0.000014,0.000049] |

The final point estimates favor coarse, but uncertainty includes no difference.
The selected arms are essentially equal and poor. No fine-resolution adoption.

## Actual execution, resources and artifacts

Separate processes: train-only smoke, main comparison, then fresh replay. Smoke
initialization/sampling/config/split/actual saved files are bound before main;
each main arm's first two updates match its smoke parameter hash exactly.
Main completed naturally; no numerical fault, resource stop or setting change.

| Process | UTC start → end (2026-10-14) | Body seconds | Peak GPU allocated | Peak process working set |
|---|---|---:|---:|---:|
| Smoke | 11:01:55.777768 → 11:02:19.469307 | 23.692 | 1,610,422,272 B | 2,303,430,656 B |
| Main | 11:02:42.804364 → 11:13:54.792448 | 671.985 | 1,555,210,752 B | 2,332,979,200 B |
| Fresh replay | 11:14:09.022098 → 11:14:40.318400 | 31.300 | 1,158,413,312 B | 1,996,333,056 B |

RTX 3080 Laptop 16GiB, Windows, Python 3.11.9, Torch 2.6.0+cu124,
CUDA 12.4, NumPy 2.4.6, FP32, TF32 off, highest matmul precision, deterministic
algorithms, four CPU threads. SDPA enablement flags are recorded in the contract;
actual kernel-level timings were not profiled. No environment upgrade or cloud.
Preflight requires 12GiB free GPU, 8GiB disk reserve, 4GiB GPU/process cap and a
3,600-second body limit checked in batches/evaluation and at completion.

Main training timers total 293.395s; epoch-evaluation timers total 339.553s.
Coarse training/evaluation: 72.591/75.706s; fine: 220.804/263.847s. These cached-feature
process timers include input handling and guards. Three CUDA-process bodies total
726.977s; do not call this pure GPU time or full native inference latency.
Independent prediction audit took 0.863s; protected-source rehash took 8.745s.
Cloud cost is zero; electricity/pure GPU time are unmeasured. Development tests,
prior feature extraction and older experiments are outside these process totals.

Both heads contain 232,385 trainable parameters; each best file is 939,250 bytes.
Logical frozen feature producers are TAPIR 31,072,327 plus cached MiniLM 17,389,824,
giving 48,694,536 with this head. This is not an all-modality resident runtime or
verified inference-package size. No producer weights are loaded/trained in this fit.
TAPIR source revision `730cda1c730877cfedbe01bf87fb1cadb78a565d`;
cached text revision `4ca70771034acceecb2e72475f72050fcdde4ddc`.

Artifacts under `C:/CodexArtifacts/pretrained-reuse-r2/`:

- `fine-qa-grouped-smoke-01/`: effective config, split, prefix weights, orders,
  predictions, optimizer/RNG state and exact reload (13,815,338 bytes).
- `fine-qa-grouped-learning-01/`: every epoch, all original probabilities/logp/cover,
  orders, gradient norms, train/selector metrics and separate best/final weights
  and optimizer/RNG states (24,174,017 bytes).
- `fine-qa-grouped-fresh-01/`: all four best/final original prediction replays
  (944,892 bytes). Saved training state is not claimed as strict-equivalent resume.
- `fine-qa-grouped-independent-audit-01.json`: independent NumPy metric/selector/
  exposure/bootstrap recomputation and exact shape/dtype/array-byte replay check.
- `fine-qa-grouped-protected-sources-01.json`: 835 unique source/code files verified,
  covering the previous 420 source pins and the main 803 input pins.

Main config SHA: `a6d303a03cb5e2c494c56e89d3768a8fa7154ae6a3cd01ff0a1ec9e2844b07e1`.
Main result SHA: `1ee6977e8e629939425c5f19e68fbe3c4924c15c82a167dbc43df2557bbbffa7`.
Fresh result SHA: `19c0ed3b0a60e44c4ded53518e99136cded52b9f4d524d7b380a99a3278554f2`.
Independent audit SHA: `9664565ac8a8380830891dd90813ed2e26b1a169081c8e2cf844919af049b504`.

Commands, run independently in this order at the pinned source commit:

```powershell
python -m scripts.train_fine_grouped_qa --smoke --output C:/CodexArtifacts/pretrained-reuse-r2/fine-qa-grouped-smoke-01
python -m scripts.train_fine_grouped_qa --smoke-from C:/CodexArtifacts/pretrained-reuse-r2/fine-qa-grouped-smoke-01 --output C:/CodexArtifacts/pretrained-reuse-r2/fine-qa-grouped-learning-01
python -m scripts.train_fine_grouped_qa --replay-from C:/CodexArtifacts/pretrained-reuse-r2/fine-qa-grouped-learning-01 --output C:/CodexArtifacts/pretrained-reuse-r2/fine-qa-grouped-fresh-01
```

These output directories now exist and must not be reused. Replay requires the
original source/config contract; running at a different commit is rejected.
The JSON records the independent auditor command inputs and all executable hashes.

## Verification and decision

Read-only review found and fixed actual config/split/reference admission omissions
before GPU execution. After main/fresh completion, a separate auditor correction
added actual replay-config/reference/terminal binding and strict dtype/signed-zero
byte comparison, with RED→GREEN regressions. No active-run source/config changed.
Independent arithmetic agrees within 1e-6; four full fit/selector replay archives
have identical keys, shapes, dtypes and array bytes. Weak source labels were reused,
not independently reconstructed by this metric auditor.

Ruff passed; full local CPU tests: 1,403 passed / 1 skipped. Dataset/model manifests,
four catalogs and ALLOW policy validation passed. Torch-absent CI coverage is
verified locally (1,001 passed / 94 skipped); publication HEAD GitHub checks are read after push and saved in
the separate publication receipt. Weights/media/large feature caches remain outside Git.

Fine resolution under this readout is **not supported as a quality improvement**.
Preserve the input corpus and both failed fits. Do not repeat this budget/seed/head
search on these IDs. The next useful gate is source-free semantic cue-to-object/
container identity and temporal memory, with diverse grouped supervision and
original-question controls before another quality fit. Finer bbox support alone
does not establish that gate. Text/image/audio/joint/OOD/release work remains open;
this result changes no four-modality target and authorizes no compression or promotion.
