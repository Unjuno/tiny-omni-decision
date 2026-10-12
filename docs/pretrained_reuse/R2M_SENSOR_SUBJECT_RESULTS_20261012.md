# Subject-disjoint sensor comparison: initial measured results

Both prescribed12epoch runs completed at source15f8e60. Training used62,269
windows from14 subjects, validation12,763 windows from3 separate subjects.
No calibration/audit subjects were read. Existing artifacts were preserved.

| Selected model | Epoch | Accuracy | Balanced Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Temporal convolution |9|0.914675|0.732640|0.241235|0.118460|0.020091|
| Static statistics |12|0.938729|0.698005|0.189363|0.087322|0.011646|

The predefined minimum validation NLL selects these checkpoints; best reload
reproduces probabilities exactly. Static statistics wins sample-weighted metrics,
but temporal has higher present-class mean recall. Neither meets95%; do not adopt
either as a general Omni result. Balanced Accuracy averages supported classes;
full support counts, including zeros, remain in the JSON. Classification labels
refer to the window center, retaining mixed windows and their ambiguity.

Fixed within-window shuffling lowers temporal Accuracy to84.6353%, NLL0.405423.
Static Accuracy is unchanged, with only floating-point reduction differences in
probabilities. This establishes order sensitivity, not superiority of temporal
reasoning: shuffling creates unrealistic signal dynamics. Static model has2,380
parameters versus6,700 temporal; the comparison is not capacity matched.

Temporal loop measured21.54seconds, static11.68seconds on local RTX3080Laptop,
float32. Whole-process peak allocated VRAM341,386,240bytes includes resident
training/validation tensors. The static peak is inherited from the shared process,
not an independently reset mode measurement. Each mode consumed747,228 training
window encounters,62,269 unique windows. Temporal selected epoch9 corresponds
to560,421 encounters; static selected epoch12 to747,228. No paid compute.

Raw histories, epoch probabilities with IDs/subjects/mixed flags, best/final
weights, config and hashes are stored separately in
`C:/CodexArtifacts/pretrained-reuse-r2/sensor-subject-training-01`.
[Summary and hashes](R2M_SENSOR_SUBJECT_RESULTS_20261012.json).
Separate training-only smoke passed before the main run. Ruff passed; CPU test
suite484passed,1skipped. No extra seeds, extended training or tuning were run.

Still required: per-subject and transition-subset analysis, exact-content overlap
audit, uncertainty at subject level, isolated runtime measurements and comparison
with a reusable pretrained sensor representation if justified. The overlapping
windows are not independent trials; only3 validation subjects are available.
These are development results with checkpoint selection, not blind evaluation.
All multimodal fusion and final product quality gates remain open.
