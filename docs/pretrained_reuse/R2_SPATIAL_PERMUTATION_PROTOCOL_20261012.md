# Fixed spatial-cell permutation diagnostic

Use frozen spatial/bilinear epoch16 from dinov2-spatial-bilinear-01. No training,
checkpoint selection, calibration or additional seeds. Evaluate identity, horizontal
4x4 cell reversal, and one seed17 permutation of16 cells. CLS remains unchanged.
Use the existing512 validation images and all2048 questions/options. These are
feature interventions, not physically flipped images: contextual tokens still
contain original encoder position information. Do not claim image invariance.

Require identity probabilities to match saved predictions within1e-5 on CPU.
Record all task metrics, paired image uncertainty, probabilities, permutations,
input/checkpoint hashes and time. This diagnoses positional dependence only;
no validation-based permutation selection or model replacement is allowed.

Execution amendment before interventions: CPU identity gate failed at1.359e-5
(threshold1e-5). Attempt01 preserved; no intervention result produced.
Attempt02 uses the original CUDA inference backend with unchanged threshold.
Attempt02 stopped before metrics: option embeddings remained on CPU while
indices were on CUDA. Attempt03 explicitly transfers options; no model/data change.
