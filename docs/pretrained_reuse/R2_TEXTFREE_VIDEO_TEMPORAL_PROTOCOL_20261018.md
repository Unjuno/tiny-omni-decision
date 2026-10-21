# Text-free CLEVRER video Decision: temporal readout comparison

**Protocol frozen before supervised fitting.** This is a narrow Path A probe of
whether a small temporal memory adds value to a pretrained video representation
for typed CLEVRER decisions. It is not a general video-language result, a
multimodal fusion test, a sealed audit, or a product-quality claim.

## Question and arms

Given frozen VideoPrism frame features, a CLEVRER typed program, a taxonomy and
the caller's answer options, can a small Decision readout return calibrated
probabilities over those options? Compare two temporal blocks with the same
feature/query path and a 1% maximum temporal-block parameter-count difference:

- `mean_mlp`: project each of the eight frame vectors to width 128, apply the
  same framewise 128→384→128 MLP independently, then mean-pool the outputs.
- `bilstm`: project the same eight frame vectors to width 128, apply one
  bidirectional LSTM with hidden width 64 in each direction, then mean-pool its
  128-wide outputs.

Both arms use the same 32-wide program-token embedding, 16-wide taxonomy
embedding, 32-wide candidate embedding and shared candidate scorer. The
vocabulary is deterministically built from training rows only. Candidate
logits are normalized only over the supplied options. There is no natural
language question encoder and no input use of question text, sample ID, scene
ID or media hash. `program` is a typed symbolic criterion, so this is not a
criterion-free video task.

For the pinned 512-row training manifest, instantiated model counts are
226,865 (`mean_mlp`) and 227,377 (`bilstm`) trainable parameters; the complete
model difference is 512 parameters (0.23%).

## Frozen data and split gates

- Source: `MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef`,
  `CC0-1.0`, admitted by `manifests/candidates/clevrer.yaml` for training.
- Sample: `C:/CodexArtifacts/pretrained-reuse-r2/clevrer-expanded-seed17-256train-200val/`.
- `train.jsonl` SHA-256:
  `5cd4e5155b0161e35b10aad2c98e14afba7d46407e635bd1c08d786e1e51765d`.
- `validation.jsonl` SHA-256:
  `53aa6f01447a43337a601279daf620c0df91d807773522cc92230fcab173db7c`.
- `fetch-report.json` SHA-256:
  `75f0511bb94df2abd7e2734c0881af0282ed3797ec7b725cc07fa7dc8fd84a44`.
- `pretraining-gate.json` SHA-256:
  `43dfa8f2c1589b6569ec75a546193cdeda63ff2e891564d9b7bf2353f5627174`.
- Train: 512 unique decisions, 256 scene groups and 256 media assets; 256
  temporal-descriptive and 256 static-descriptive rows.
- Validation: 400 decisions, 200 scene groups and 200 media assets; 200 rows
  per question type. Train/validation sample, scene, media and normalized
  question/options overlap must remain zero.
- Every media path and SHA is rechecked before fitting. The archive-level
  SHA-256 was not computed by the frozen source gate; the selected media bytes
  are individually SHA-256 verified.
- This fixed validation is development data used for checkpoint selection. It
  is not loaded from a sealed-audit store and must not be called blind.

## Frozen video features

VideoPrism-B `google/videoprism-base-f16r288` at revision
`be719a406d563b66f0ac969e7c94bab8e997c81a`, upstream source commit
`c6eba0ece3dd49a24ffc44582966997a4a59f1d8`, and checkpoint SHA-256
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
Extract eight endpoint-inclusive frames at 288×288 and average spatial tokens
within each output frame to obtain `[8,768]` float32 per unique video. The old
framewise cache contains the 256 train videos but a different validation set;
it is not used as the validation feature source. A fresh immutable cache is
made from the exact frozen train/validation manifests. No encoder is updated.

## Fit and selection

Both arms use seed 17, the exact same row-wise epoch permutations, batch size
64, 256 epochs (8 updates/epoch, 2,048 optimizer updates), AdamW at `1e-3`,
weight decay `1e-4`, constant learning rate and gradient norm clip 1.0. Epoch
orders are generated as `numpy.default_rng(17 + zero_based_epoch).permutation`
over the frozen train-row order. Every epoch consumes all 512 training rows
once. No early stopping or hyperparameter search is permitted. Each arm starts
from fresh randomly initialized readout weights; the common embeddings/scorer
receive the same seed-initialized values.

At every epoch, record training CE and train/validation Accuracy, NLL, Brier,
ECE-15 and mean confidence overall, by question type, and by question-type ×
taxonomy groups. The selector is the mean of the temporal-descriptive and
static-descriptive macro NLLs; within each question type, available taxonomy
groups receive equal weight. Ties keep the earlier epoch. Save selected and
final checkpoints separately, and verify exact validation-logit equality
after reloading the selected checkpoint.

The report separately records full-run example exposures and selected-step
exposures, unique decisions/assets/scenes, repetition counts, per-epoch
gradient norm, wall/fit/evaluation time, peak allocated VRAM and all artifact
hashes. The complete training split is 512 unique decisions; repeated epochs
are not counted as more unique data.

## Video-use controls and uncertainty

After validation selects each arm, evaluate the same examples with (1) aligned
video, (2) zeroed video, (3) video features cyclically reassigned across
validation scene groups with a fixed seed, and (4) the eight saved feature
positions reversed. These are fixed diagnostic interventions, not additional
checkpoint selectors. Store per-example probabilities for the aligned and
intervened predictions.

Compare selected arms and aligned-versus-scene-shuffled predictions using
paired scene-group bootstrap: 5,000 resamples, seed 17, percentile 95% intervals
using NumPy's linear quantile. Report Accuracy, NLL and Brier differences for
both question types. Validation checkpoint selection and the single training
seed limit the interpretation; intervals are not blind-audit inference.

## Interpretation boundary

This experiment can tell whether this readout uses this frozen video signal
and whether a matched recurrent block helps these two CLEVRER descriptive
tasks. It cannot establish open-ended video understanding, causal/streaming
behavior, AV complementarity, 95–98% product quality, or a general ceiling.
VideoPrism frame outputs are clip-contextualized; reversing their order is only
a readout-order diagnostic. The result does not authorize adding the adapter
to the complete resident model or changing its 200M pre-quantization budget.

Never write media, encoder checkpoints or trained readouts to Git. Keep all run
outputs below the new external artifact directory specified in the command.
