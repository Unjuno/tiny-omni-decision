# Data and training pipeline audit before further optimization

User priority: inspect data and the actual training path before another fit.
The completed 32,768-step curve and every frozen artifact remain unchanged.
This is a read-only development-data audit, with zero optimizer updates and no
calibration or sealed-audit payload access. No new architecture is selected here.

Trace source annotations -> questions/options/targets -> cache identities/offsets
-> runtime tensors -> shuffled cross-entropy and gradient accumulation -> metrics.
Recompute structured text labels and STARSS window labels independently. Report
unverified subjective labels as UNKNOWN. Check all corpus rows and consumed
counts separately; a source containing a label does not mean the sampler used it.
Audit complete feature bank hashes/shapes/finiteness, query association, duplicate
inputs with conflicting labels, runtime transport, video frame sampling/crop
geometry and train/validation support. CPU-only model gradient checks use copied
initial weights and do not update or save any model.

Distinguish corruption from task coverage and irreversibly discarded information.
An exactly replayable cache can still omit relevant visual/audio/text information.
An annotation match does not establish human label correctness. Any detected
limitation is retained in results; no difficult row or class is removed.

Changes to data, preprocessing, sampling, regularization or architecture require
separate versioned experiments after this audit. Keep the existing long-budget
baseline; do not infer a model-family ceiling or rule out delayed generalization.
