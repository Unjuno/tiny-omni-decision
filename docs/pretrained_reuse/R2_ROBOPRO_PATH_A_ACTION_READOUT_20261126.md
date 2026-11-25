# RoboPro frozen Path A action readout — 2026-11-26

## Result and scope

**Status: completed bounded development comparison; RoboPro is still not admitted as a general R2-M/product-quality benchmark.** The task is four-choice retrieval of one recorded demonstrator action from a synchronized camera window, robot state, and (optionally) task text. The recorded action is the target label; it is not a label for task success, safety, or optimality, and other candidate actions may also be valid.

The single seed-17 comparison used 64 train and 64 development episodes, three causal query points per episode, and 16 exact-prompt groups in each split (192 queries per split). The project-created split has no exact-prompt overlap. Physical scene/base-task independence is unknown; the publisher supplies no official validation/test split. The result is exploratory and must not be generalized beyond this candidate protocol.

The main finding is that **VideoPrism features did not improve the selected text+state readout on this small development set**. Text+state was the strongest point estimate. State was clearly more useful than the tested video-only path. This does not establish a limit of VideoPrism, recurrent video features, or video generally.

## Frozen inputs and reproducibility

- Dataset: `Hoshipu/RoboPro` revision `90ec789bf4018eb9c0f75da9f69aab5c185f0fd0`, publisher-declared CC BY 4.0. The release has no official validation/test split; the exact-prompt split and label caveats are documented in [the source/task screen](R2_ROBOPRO_SOURCE_TASK_SCREEN_20261123.md) and [the action contract](R2_ROBOPRO_PROMPT_SPLIT_AND_ACTION_CONTRACT_20261123.md).
- Project sample manifest SHA-256: `eeab697bc4189ba2b4442ed53def960f248cf146b2cc947c0633905a41340686`.
- Data audit SHA-256: `3c4742411d895972dd2a3ec905db45133697516360723b1703555e3af23ab8ae`.
- Effective config: `configs/pretrained_reuse/robopro_action_probe_v1.json`, SHA-256 `c1df84707b4ab52db3e333d8888d94d400c801424134f6f0540b800fac04d675`, seed 17. Eight arms, 80-epoch maximum, per-arm development macro-NLL selection, early stopping patience 12, train-only feature normalization, and 10,000 paired prompt-group bootstrap replicates were fixed in the config.
- Video encoder: frozen Google VideoPrism-B f16/r288, revision `be719a406d563b66f0ac969e7c94bab8e997c81a`, 114,365,184 parameters; checkpoint SHA-256 `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`. Inputs use eight frames over at most two seconds ending at each query time. JAX was forced to CPU.
- Text encoder: frozen `sentence-transformers/paraphrase-MiniLM-L3-v2`, revision `4ca70771034acceecb2e72475f72050fcdde4ddc`, 17,389,824 parameters; weights SHA-256 `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`. Prompt embeddings use an attention-mask mean of last hidden states, max length 256. Torch was forced to CPU.
- The readout is trainable; all pretrained encoders are frozen. The full text+state+Video-GRU arm has 131,989,633 pre-quantization parameters, including both frozen towers and the readout. Its parameter count is within the provisional 100–200M target; no quantized package, package bytes, or target-device latency was measured.
- Code present at execution was based on source worktree HEAD `12d63cad04e07ae53ba6c139f7186731668a5267` plus then-uncommitted RoboPro probe files. Their bytes are preserved in this branch. Key execution-source SHA-256 values: feature extractor `8006205ad8dff00b14d1f2378381f3c26fd6b86f3d284c6288784db71496b4cf`; text extractor `d1e3654e3b6b864dd0c38e1e601503a6d6571dd3411f0afa111c98916a21e150`; readout fitter `fb2374296a03c888b6c2eacdadfc419f4557cc95ee92b3d59122da536a6ca8a8`.
- Environment: Windows, Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2, JAX 0.6.2, Flax 0.11.1, TensorFlow 2.18.1, PyAV 18.1.0. Feature extraction and fitting used CPU; no GPU or cloud compute was used. Video extraction took 480.61 s (424.02 s forward); text extraction 28.28 s; all eight readout fits and controls 12.30 s. The readout process peak working set was 585,764,864 bytes. Cloud cost: 0.

Exact commands (from repository root; paths are the recorded local paths):

```powershell
& 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe' `
  scripts/extract_robopro_probe_features.py `
  --sample-manifest manifests/pretrained_reuse/robopro-action-probe-sample-20261123.json `
  --data-audit 'C:\CodexArtifacts\pretrained-reuse-r2-local\robopro-action-probe-v1-attempt-02\data-audit.json' `
  --data-root 'C:\CodexArtifacts\pretrained-reuse-r2-local\robopro-action-probe-v1-attempt-02' `
  --parquet-arrays 'C:\CodexArtifacts\pretrained-reuse-r2-local\robopro-action-probe-v1-attempt-02\parquet-array-sidecar.npz' `
  --checkpoint 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-f16r288.npz' `
  --source-root 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source' `
  --output-root 'C:\CodexArtifacts\pretrained-reuse-r2-local\robopro-action-probe-v1-attempt-03\video-features'

python scripts/extract_robopro_prompt_features.py `
  --sample-manifest manifests/pretrained_reuse/robopro-action-probe-sample-20261123.json `
  --model-dir 'C:\CodexArtifacts\pretrained-reuse-r2-local\robopro-action-probe-v1-attempt-02\minilm-l3-model' `
  --output-root 'C:\CodexArtifacts\pretrained-reuse-r2-local\robopro-action-probe-v1-attempt-03\text-features'

python scripts/fit_robopro_action_readouts.py `
  --config configs/pretrained_reuse/robopro_action_probe_v1.json `
  --video-features 'C:\CodexArtifacts\pretrained-reuse-r2-local\robopro-action-probe-v1-attempt-03\video-features\robopro-video-state-features.npz' `
  --text-features 'C:\CodexArtifacts\pretrained-reuse-r2-local\robopro-action-probe-v1-attempt-03\text-features\minilm-prompt-features.npz' `
  --output-root 'C:\CodexArtifacts\pretrained-reuse-r2-local\robopro-action-probe-v1-attempt-03\readout-run-01'
```

The feature reports are under `C:\CodexArtifacts\pretrained-reuse-r2-local\robopro-action-probe-v1-attempt-03`. They record VideoPrism feature-cache SHA-256 `edf29bf7fda80b5fcc03030f28342fb7b4bacbefc3c911078e34a467f956dd10` and text-cache SHA-256 `3ed4a8643eea81761b8140da33c74ea3ad11437c9dfafb2fd75fb5e247990bba`. The readout run report SHA-256 is `41107a5258f6475295e8993d8fab711eb250d9b28f4aea7b22ae5e5a99bd759a`; per-example prediction SHA-256 is `56de250d1b4e6cde6d0c81cb760531d1796ebb2fd636cea2adac82a49511261f`; training-history SHA-256 is `5c7e3d25f9231972e39dc37a0c0d598e0892920afc324e9d30457758f31c1949`.

Each arm's selected checkpoint was loaded from disk and rescored; all eight maximum probability deltas were exactly `0.0` at the stored float32 precision. The eight checkpoint SHA-256 values are recorded in the external run report and remain outside Git with the rest of the run artifacts.

## Development metrics

Metrics are prompt-group macro values (each prompt has 12 queries); 192 individual queries are nested in only 16 development prompt groups. `Best` is the per-arm epoch selected by minimum development macro NLL, so the displayed development metrics are selection-exposed.

| Input arm | Best epoch | Total resident parameters | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|---:|
| Text | 6 | 17,440,577 | 0.2500 | 1.3866 | 0.7502 | 0.0065 |
| State | 45 | 33,345 | 0.5573 | 0.8461 | 0.4838 | 0.1474 |
| Video mean | 5 | 114,440,513 | 0.3542 | 1.3604 | 0.7378 | 0.1715 |
| Video GRU | 8 | 114,551,297 | 0.3125 | 1.3562 | 0.7379 | 0.1915 |
| Text + State | 55 | 17,456,129 | **0.5833** | **0.8356** | **0.4954** | 0.1917 |
| Text + Video mean | 6 | 131,863,297 | 0.2917 | 1.3436 | 0.7297 | 0.1582 |
| Text + State + Video mean | 29 | 131,878,849 | 0.5260 | 0.9363 | 0.5262 | 0.1864 |
| Text + State + Video GRU | 21 | 131,989,633 | 0.5156 | 0.9089 | 0.5364 | 0.1808 |

The lower ECE for some near-uniform arms is not evidence of useful calibration; their accuracy is at or near four-choice chance (0.25). Text+State is the best point estimate but also has a substantial train/development gap (train macro Accuracy 0.7396 vs 0.5833; train NLL 0.5145 vs 0.8356).

The config's paired prompt-group bootstrap compared the selected Text+State+Video-GRU arm with State-only: accuracy difference `-0.0417`, 95% percentile interval `[-0.0990, 0.0208]` (10,000 resamples, seed 17). A secondary paired bootstrap on the saved predictions compared the selected joint arm against Text+State: macro Accuracy difference `-0.0677`, 95% percentile interval `[-0.1510, 0.0156]`; macro NLL difference `+0.0733`, interval `[-0.0409, 0.1921]`. This secondary analysis was performed after the fixed run and is descriptive; both intervals are wide and include zero.

On the selected joint checkpoint, zeroing state reduced macro Accuracy to 0.2760 and NLL worsened to 1.9736; misaligning state reduced Accuracy to 0.1563 and NLL worsened to 3.1365. Zeroing the video sequence changed Accuracy from 0.5156 to 0.5104 and NLL from 0.9089 to 0.9180; shuffling video windows within prompt/progress reduced Accuracy to 0.4792. These are fixed-checkpoint sensitivity controls on the same 16 development prompt groups, not independent confirmation that video is useful.

Candidate-order permutation changed no argmax predictions; maximum probability change was `1.19e-7`. The report preserves every selected-arm development prediction, target and probability. No official test, sealed audit, model promotion or quantization was performed.

## Interpretation and next gate

- The current BERT-family evidence remains task-specific: the four-layer LINE DistilBERT beat the matched Japanese text baseline on KUCI, while its common four-modality fit did not beat the native readout. This RoboPro run used MiniLM-L3 and is not a BERT-series comparison.
- Do not run a broad BERT sweep on exposed corpora. If a new independent text task is admitted, compare one parameter-budgeted BERT-family encoder with a matched alternative under a fixed protocol.
- For this synchronized task, the immediate bottleneck is weak video-only transfer and sparse independent prompt support, while state has a large signal. The next justified step is not more video temporal-head tuning on these 16 prompts. First establish a larger, scene/base-task-grouped sample and adjudicate whether different recorded actions can all be valid. Until that is done, this result is a pipeline diagnostic, not an R2-M quality pass.
- R0 device/task battery, broad independent task quality, audio/image/video product gates, quantized package/runtime, and R8 locked evaluation remain open.
