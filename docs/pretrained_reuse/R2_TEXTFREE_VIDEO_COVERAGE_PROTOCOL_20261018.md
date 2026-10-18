# Text-free CLEVRER unique-scene coverage comparison

Status: pre-registered development experiment; no results are claimed here.

## Question and scope

Test whether doubling unique train scenes from 256 to 512 helps a small
text-free video Decision readout when the frozen VideoPrism encoder,
architecture, optimizer-update budget and total example presentations are
held fixed. This isolates train-scene coverage more directly than the earlier
comparison, which changed the number of presentations along with coverage.

This is a single-seed (17), development-only experiment. It is not a blind
audit, does not establish a product capability, and does not change any
multimodal product gate. Natural-language question text and scene identifiers
are excluded from model inputs. CLEVRER question templates repeat across the
split; the new sample explicitly reports this content overlap while keeping
scene, media and sample identity disjoint. Validation is therefore suitable
for a controlled development comparison but is not evidence of unseen-template
generalization.

## Frozen inputs and arms

- Source: CC0 CLEVRER revision
  `98b842082ba4f7c18b6b9e3f39145871782a65ef`.
- Train: 1,024 examples from 512 videos/scenes, half static descriptive and
  half temporal descriptive.
- Validation: 512 examples from 256 different videos/scenes, half static and
  half temporal descriptive.
- Both splits use the same five CLEVRER taxonomies. Video, scene, and example
  identities are disjoint. Repeated question templates/content are counted
  and reported, not silently treated as independent content.
- Frozen VideoPrism base f16r288, eight endpoint-inclusive frames, framewise
  spatial-token mean pooling, official pinned source and checkpoint hashes.
- Two readouts only: `mean_mlp` and `bilstm`; same seed, input features,
  optimizer, learning rate, weight decay, gradient clipping and selector.
- One example per item, batch size 64; 128 epochs × 16 updates = 2,048
  optimizer updates and 131,072 example presentations per arm. No early stop.
- Validate each epoch. Select the earliest checkpoint attaining minimum macro
  question-type NLL (mean of temporal and static descriptive NLL).
- Report Accuracy/NLL/Brier/ECE by modality-equivalent video question type,
  aggregate and per taxonomy, plus training/validation curves and paired
  scene-cluster bootstrap intervals (5,000 resamples, seed 17, percentile
  95% interval). Keep selected and final checkpoints distinct.

## Integrity and execution controls

The sample manifests, fetch receipt, media-integrity receipt, encoder source,
model revision and checkpoint are pinned in the JSON config. Before feature
extraction, verify all media hashes, split grouping, expected counts, source
license and zero scene/media overlap. Extract one feature vector sequence per
unique video; preserve the resulting cache and provenance outside Git. Before
fitting, record the effective config and hash, cache hash, exact ordered sample
ID hashes, split counts, content overlap, runtime versions, device, and output
path. Run tests and a separate CPU smoke process before the full fit.

All work is local and CPU-only so it cannot contend with the separately
supervised Orbitune GPU process. No cloud compute, official test, sealed audit,
additional seed, or extra training budget is authorized by this protocol.
Existing datasets, caches, checkpoints, and artifacts remain read-only. New
outputs use a fresh attempt directory and must not enter Git if they contain
weights or media.

## Interpretation fixed before results

Coverage is supported only if the larger unique-scene arm improves validation
macro question-type NLL and its paired temporal metrics without a material
tradeoff hidden by aggregate Accuracy. A mixed result is uncertain. A null or
negative result does not establish a model-family ceiling; it only says that
unique-scene doubling did not show a benefit under this encoder, sample,
single seed and fixed budget. Repeated templates, one seed, shared validation
selection, and frozen visual representation limit causal and generalization
claims. No arm is promoted to the product model on these results alone.
