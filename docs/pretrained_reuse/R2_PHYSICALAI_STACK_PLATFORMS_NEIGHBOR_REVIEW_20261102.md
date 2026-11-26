# PhysicalAI stack-platforms top visual-neighbor review — 2026-11-02

## Scope and result

This is a label-blind visual review of the eight highest train/validation
episode pairs from the already recorded full-split DINOv2 embedding-neighbor
screen. It does not reload a model, extract features, inspect action labels,
change a split, refit a readout, or evaluate quality.

Across the sampled pairs, the three fixed temporal samples per episode show a
shared camera, robot, work surface, and simulated scene geometry. Object
positions and colors vary between episodes. This confirms that these nearest
pairs are not evidence of scene-held-out evaluation: the episode split leaves
the visual environment/background common across train and validation. The
review does **not** establish exact duplicate trajectories, identify every
scene in the 950-episode pilot, or prove the source has only one scene.

This strengthens the existing decision to keep the action pilot as
`REVIEW_NOT_ADMITTED` and to describe its validation only as same-source,
episode-held development evidence. Do not tune or refit the pilot on its
observed validation IDs. The source metadata still has no publisher scene,
layout, or environment identifier.

## Reproduction inputs

- Source revision: `73c00c18e7ca108e91846294b20a1d7cd5de9340`.
- Encoder: DINOv2-base
  `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, FP32.
- Neighbor source JSON: `C:/CodexArtifacts/pretrained-reuse-r2/physicalai-stack-platforms-episode-similarity-20261102-attempt-01/episode-similarity.json`, SHA-256
  `ea5ec427a8d659c050a593c648128237933603d261cd3f2dd5122fb77ebe6023`.
- It records 175,000 train/validation pairs, 78 pairs with cosine >=0.98,
  and maximum cosine 0.9900816. These thresholds are descriptive, not a
  validated duplicate rule.
- The contact sheet uses the top eight pairs and frames near 20%, 50%, and 80%
  of each source video. It contains no labels or model predictions and is
  stored outside Git at
  `C:/CodexArtifacts/pretrained-reuse-r2/physicalai-stack-platforms-top-neighbor-review-20261102.png`,
  SHA-256
  `fd40a1b0a392ea61071c92e6fc90879b7ea9eac049e30a1ce9487e43e9e96516`.
- Train/validation video bytes were read from the already downloaded, hash-
  verified 700/250 episode pilot. No reserved episode was accessed. This was
  CPU-only and incurred no cloud cost.

The audit is a descriptive follow-up, not a replacement group manifest and not
a causal explanation of the measured pilot scores. A future admission attempt
needs publisher scene/layout identity or a separately frozen source with
auditable scene groups.
