# Frozen trajectory Decision comparison — protocol

The previous pooled-window heads did not demonstrate visual grounding. This
comparison tests ordered point trajectories with the tracker's own initial
appearance features, using the existing 99 train / 257 development-validation
videos and all 198 / 514 questions. It is not a blind or sealed evaluation.

## Fixed extraction

- Pinned legacy TAPIR weights and strict constructor from the runtime audit.
- Frozen FP32, TF32 disabled, RGB256, all frames before the exclusive reveal
  cut; -1 means the complete clip. No per-video truncation or dropped failures.
- Uniform16×16 queries at frame0; query chunk16. No annotation-selected queries.
-32 uniformly spaced trajectory times: xy and two uncertainty channels.
-384 initial appearance values from the existing TAPIR query descriptor.
-256×512 tokens per clip, with media/source/feature hashes and actual frame IDs.
- Existing source media manifests and pair hashes remain fixed. Verify media
  identity/content disjointness before training. No sealed data is loaded.
- Extraction errors stop the experiment and are retained; do not discard clips
  based on tracking quality or labels. All356 feature files required for fitting.

## Fixed readout comparison

Use the new164,032-parameter TrajectoryReadout, frozen cached MiniLM candidate
features, seed17, AdamW lr0.001 / weight_decay0.01, batches of8 video pairs,
50 epochs. Keep the same initial/final question pairing and choose the earliest
minimum development-validation NLL, following the previous head experiments.
Save untrained, best and final predictions/checkpoints and every epoch.

Evaluate full tokens, cyclically mismatched videos and zero tokens. Report
Accuracy/NLL/Brier/ECE, initial/final metrics, both-correct and changed-target
pairs. Report train/validation curves and paired video bootstrap differences
against the saved prior heads. No tuning on the resulting comparison, no seed
search or removal of difficult videos. Single-clip smoke is excluded from model
selection; initialize the full run afresh. No encoder updates or compression.

Geometry and initial appearance may still fail to identify the hidden object.
A better aggregate score without mismatch sensitivity or changed-condition
success is insufficient evidence to adopt the representation. This experiment
addresses only the video Decision bottleneck; other modality and release gates
remain open.
