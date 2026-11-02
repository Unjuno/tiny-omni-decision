# PhysicalAI stack-platforms action-prediction pilot — 2026-11-02

## Status and scope

The frozen R2 pilot protocol was executed once on the 700-train / 250-validation
episode split. This is a same-source, episode-held development experiment, not
an independent source or scene-held evaluation. The source remains
`REVIEW_NOT_ADMITTED`: publisher metadata does not identify scene/layout
identity, semantic near-duplicate overlap has not been audited, and exact
normalized-frame hashing cannot establish scene independence. No reserved or
sealed holdout was accessed. These results do not establish real-world task
success, optimal behavior, or product readiness.

The fixed target is the demonstrator's 7-D action three 30-fps frames (100 ms)
after each query. Eight train-only action prototypes define an 8-way
classification task. The four query windows per episode use eight sampled
frames, 81-D proprioceptive history, and frozen FP32 DINOv2-base features.
Linear softmax readouts use train-only feature normalization and LBFGS
(`max_iter=200`, seed 17). The primary comparison is the joint video + state
readout; controls are descriptive and were fixed before fitting. The 250
validation episodes / 1,000 queries were evaluated once and were not used for
model selection.

## Implementation and run

- Source commit: `2c4cf97f0eef3bec3e511fdc0f7452d3b0b9b984`.
- Frozen split manifest v3 SHA-256:
  `3a2849afaa3cefa414c9dbfc094976e0f1cc48ad0e9636d65573fa306d1ed967`.
- Extraction command:
  `python scripts/extract_physicalai_action_pilot.py --artifact-dir C:\\CodexArtifacts\\pretrained-reuse-r2\\physicalai-stack-platforms-pilot-features-20261102-attempt-01 --model-dir C:\\CodexArtifacts\\pretrained-reuse-r2\\dinov2-base-runtime-smoke-01\\model --device cuda --batch-size 4`.
- Evaluation command:
  `python scripts/evaluate_physicalai_action_pilot.py --artifact-dir C:\\CodexArtifacts\\pretrained-reuse-r2\\physicalai-stack-platforms-pilot-features-20261102-attempt-01`.
- Extraction started 2026-11-02 14:47:50 JST and completed in 590.96 s
  (about 9 min 51 s). It processed exactly 3,800 queries: 2,800 train and
  1,000 validation, 8 frames/query, 700/250 distinct episodes, four windows
  per episode. The live progress log reached 950/950; output counts and
  timestamp checks passed. The evaluator completed successfully and verified
  saved-checkpoint reload probabilities exactly (max absolute delta 0).
- Extraction measured 179.26 s video decode and 404.42 s DINO inference;
  peak PyTorch CUDA allocation was 402,263,040 bytes. The RTX 3080 Laptop GPU
  was shared, with utilization varying over time; these are observed wall and
  allocation measurements, not isolated device benchmarks. Evaluation and
  bootstrap ran on CPU. No cloud compute or charge was used.
- Runtime/model revisions, hardware, package versions, FP32 settings, and
  model-file hashes are recorded in `feature-extraction.json`; see the runtime
  smoke report for environment details. No pre-existing corpus, checkpoint,
  Teacher artifact, or reserved payload was modified.

## Validation results

All metrics are on the same 1,000 fixed validation queries. ECE uses 10 equal
confidence bins. Lower NLL/Brier/ECE is better; higher accuracy is better.

| Readout / control | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Proprio history | 0.704 | 5.413 | 0.591 | 0.006 |
| Frozen video, 8-frame mean | 0.584 | 7.342 | 0.827 | 0.051 |
| Frozen video, last frame | 0.506 | 8.583 | 0.982 | 0.093 |
| Joint video + proprio | **0.725** | **4.490** | **0.543** | 0.077 |
| Joint readout, video from next episode | 0.548 | 7.663 | 0.888 | 0.100 |
| Joint readout, video input missing | 0.632 | 5.897 | 0.725 | 0.113 |
| Joint readout, proprio input missing | 0.321 | 11.092 | 1.323 | 0.281 |
| Train class prior only | 0.356 | 1.894 | 0.812 | 0.023 |

All four fitted readouts reached 100% training accuracy with effectively zero
training NLL/Brier/ECE (2,800 queries), while their validation scores were
substantially lower. This is direct evidence of severe readout overfit under
the fixed protocol. For example, joint video+proprio moved from train 100.0%
Accuracy / 0.000 NLL to validation 72.5% / 4.490 NLL. No regularization or
hyperparameter changes were tried after seeing this result.

The action-prototype validation support was imbalanced: class counts were
`[78, 74, 106, 92, 69, 137, 88, 356]` (train counts
`[206, 255, 260, 217, 231, 354, 216, 1061]`). The high NLLs and the strong
class-prior baseline expose a skewed/easy-to-game target distribution; accuracy
alone is not a sufficient quality measure. Joint input improved validation
accuracy by 2.1 points over proprio-only, but the episode-cluster bootstrap
95% interval crossed zero: mean `+0.021`, interval `[-0.008, +0.051]`.
Joint input exceeded the video-only 8-frame-mean accuracy by 14.1 points
(interval `[+0.106, +0.177]`). Replacing video with the next episode reduced
joint accuracy by 17.6 points (paired interval for joint minus misaligned
`[+0.142, +0.212]`); removing video reduced it by 9.3 points
(`[+0.062, +0.123]`). Removing proprio reduced it by 40.3 points
(`[+0.367, +0.439]`). Bootstrap used 2,000 episode-cluster resamples, seed 17.
These are diagnostics on the same validation set used for the reported
comparison, not an untouched inferential test.

After the one-shot metrics were saved, the existing representation-neighbor
screen was run over the same frozen features as a split-integrity diagnostic,
without reading action labels or changing the split/readout. Among all 175,000
train/validation episode pairs, the maximum cosine similarity of normalized
episode-mean DINO features was `0.99008`; 78 pairs were at least `0.98`, and
one was at least `0.99`. The nearest pair was validation episode 12627 and
train episode 12613 (episode-index distance 14). These are only embedding
neighbors: the thresholds are descriptive, not validated duplicate cutoffs,
and this screen cannot establish that those episodes share a scene or are
duplicates. It strengthens the case to keep the source unadmitted until
publisher scene/layout identifiers or a defensible scene-level audit exist.
Since the task labels were already inspected, this post-hoc screen is not used
for model selection or to revise the reported metrics.

## Interpretation and next gate

The result is consistent with both inputs contributing to held-out-episode
prediction of this dataset's demonstrator actions; the shuffled-video control
suggests the video embedding carries episode-aligned signal. It does not show
that the model learned generally valid temporal reasoning: the linear probe
uses aggregate frozen frame features, validation episodes may share scenes or
layouts, and the target is imitation of the dataset's action labels rather
than task reward or success. Joint performance over proprio alone is modest
and statistically uncertain under the episode bootstrap, while calibration
and class imbalance need attention in a newly specified protocol.

Do not admit this source or promote the result to a product-quality claim.
Before further modeling, obtain publisher-level scene/layout identifiers or
another source with auditable group identity, perform semantic duplicate
checks, and define a less skewed task target tied to success or counterfactual
decision quality. Any follow-up must freeze a new protocol and split before
examining its validation labels. The current pilot need not be rerun or tuned.

## Artifact integrity

Artifacts are outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2/physicalai-stack-platforms-pilot-features-20261102-attempt-01/`.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `features.npz` | 28,569,116 | `235660d6a8947f7c0a9908541e5415cc5db36149a334e6baa7d0d32b347d4ef9` |
| `feature-extraction.json` | 1,094 | `2484a1a448464693d08b8cc99ba827f6355810a5450beeafc0a8e1f598414111` |
| `pilot-metrics.json` | 15,844 | `e7ce5b7787ac27f7d3809f9d7fb3191368a5292fe115b3c04d819e852b517694` |
| `validation-predictions.json` | 2,021,438 | `3f69c09963c6515e138b581d35be66002b0362ac770391a6aed7b7c6bf8e21a2` |
| `probe-checkpoints.pt` | 150,595 | `d8e97f5040001e365b6031a97b471aabb483082cb1284d91d98fc1f3165b9f72` |

Full per-class metrics, prediction probabilities, checkpoint details, and
protocol fields are in the immutable JSON artifacts above. The baseline and
source admission status remain unchanged.

The separate similarity screen is at
`C:/CodexArtifacts/pretrained-reuse-r2/physicalai-stack-platforms-episode-similarity-20261102-attempt-01/episode-similarity.json`
(SHA-256 `ea5ec427a8d659c050a593c648128237933603d261cd3f2dd5122fb77ebe6023`).
