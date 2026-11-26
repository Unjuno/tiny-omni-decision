# Physical-mirror training feature extraction

Execution PASS: all8424 pinned training images physically mirrored, unchanged
frozen DINOv2-small and224x224 whole-image preprocessing. Every original media
SHA checked. First64 features reproduce the prior diagnostic exactly. No
validation images, sealed audit, training, calibration or checkpoint changes.

- FP32 shape[8424,17,384],219,967,488 payload bytes, finite throughout.
- Feature SHA256: `c4d7c86a5bda8d11e95cdf0927c65e686e962e2010f54585e081a08acf33ea9f`.
- 8424 original assets and8424 augmented views; zero new independent assets.
-99.127seconds on RTX3080Laptop,292,862,464 peak allocated GPU bytes,cloud$0.
- Remaining disk1,338,519,552bytes; existing artifacts retained.
- Artifact: `C:/CodexArtifacts/pretrained-reuse-r2/dinov2-train-mirror-cache-01`.
  config.json records exact model files, code hash, media order and provenance.

These are features for the preregistered matched augmentation comparison, not
new quality evidence. Training/smoke/comparison remain unexecuted at this point.
No model improvement is claimed from extraction alone.

Ruff passes;394CPUtests pass,1skip. No feature arrays or weights added to Git.
