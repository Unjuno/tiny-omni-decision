# Independent AV temporal versus conditional readout design

## Intent

Compare a temporal LSTM readout with a query-conditioned bidirectional
audio/video cross-attention readout and learned gate on
a newly generated, video-required sequence-matching task. Existing AV validation
groups are exposed, and the public natural-video candidates reviewed so far do
not have a clean combination of media rights, unused group-disjoint data, and
verified audio-plus-video necessity. A controlled synthetic task is therefore
the least-confounded architecture diagnostic available locally.

## Goals

- The answer must depend on both an eight-step visual sequence and its aligned
  audio sequence. Each modality alone, with the query, must have balanced labels.
- Compare a temporal LSTM and a conditional fusion readout using the same frozen
  VideoPrism and EfficientAT feature sequences, data order, seed, optimizer,
  update budget, and validation IDs.
- Save exact split, sequence, media, feature, config, checkpoint, and prediction
  hashes. Report Accuracy, NLL, Brier, ECE, and matched-pair accuracy.
- Measure audio-only, video-only, missing-input, shuffled-pair, and reversed-time
  controls without selecting on the controls.

## Non-goals and anti-goals

- This is not a natural-media, speech-understanding, or product-quality result.
- Do not use CLEVRER, Perception Test, HoloAssist, or other previously observed
  evaluation records to train, tune, or claim independent transfer.
- Do not unfreeze either encoder, change sequence length, change the dataset
  after seeing validation results, or tune multiple seeds against the same
  validation cohort.
- Do not claim the 95–98% product target from generated data.

## Approaches considered

1. **Reuse an exposed public AV validation split.** Rejected because prior
   validation exposure would make architecture selection circular.
2. **Wait for a newly rights-cleared natural AV source.** Best for external
   validity, but no such unused, fully admitted source is currently available;
   this would not test the already-approved architecture comparison now.
3. **Generate a controlled AV sequence task from scratch.** Selected for a
   bounded architecture diagnostic. Every visual symbol has a corresponding
   acoustic code; matched and permuted sequences use paired examples so that
   neither stream alone predicts the answer. New sequence identities are
   group-disjoint between train and validation.

## Experiment

Generate eight-step videos of four visually distinct symbols and synchronized
eight-note audio codes. A query asks whether the audio and video codes have the
same order or different order. Each underlying AV pair is presented with both
query polarities. Construct matched and same-symbol-multiset permuted pairs in
balanced groups. Keep every sequence/media asset in one split only. Verify
before fitting that audio-only and video-only label distributions are balanced
within query and group, no normalized sequence or asset hash crosses splits,
and controls have the expected chance behavior.

Use the repository's pinned frozen VideoPrism and EfficientAT encoders. Produce
one spatially pooled VideoPrism feature per each of the eight ordered frames,
and one EfficientAT feature per corresponding audio time segment, yielding two
aligned length-eight sequences. The temporal LSTM consumes both sequences and
the query embedding at each time step. The conditional readout injects the
query into each ordered sequence, performs two layers of bidirectional
audio-to-video and video-to-audio cross-attention, query-pools each updated
stream, then applies a learned two-way gate and shared binary scorer. This
operator was selected over independent query pooling because the task requires
a cross-stream comparison; it makes direct sequence interaction explicit.
Parameter-match the two readouts
as closely as practical and record exact counts. Use seed 17 and one fixed
training budget, with validation checkpoints on a preregistered schedule and
best selection by macro NLL. Save the terminal checkpoint separately. Apply the
same controls to both selected checkpoints. Any encoder or generator
precondition failure ends the experiment at that stage and is documented; do
not silently switch encoders or simplify the task.

All GPU work goes through the cooperative HotUGPU guard. It may wait while
Orbitune is using the device, but must not stop or alter that process. All new
raw data, features, and weights live under a new external attempt directory;
only small scripts, tests, configs, and summaries belong in Git.

## Success and interpretation

Success means a reproducible controlled comparison with passing identity and
input-necessity gates, not a required accuracy threshold. Report whether either
readout uses both modalities and temporal order on this synthetic task, the
validation learning curves, uncertainty clustered by generated sequence group,
and parameter/runtime cost. If both fail, keep the result and diagnose the
frozen features or task encoding. A positive result only justifies designing a
new natural AV source study; it does not close R2 M-2 or qualify the product.

## Implementation calls

- Default: generated symbolic video plus tone sequences, because no new
  rights-cleared natural AV cohort is currently admitted.
- Split unit: complete sequence-pair group, with unique per-split sequence and
  asset hashes.
- Model selection: fixed seed, common update budget, validation NLL only.
- Compute: local hardware through HotUGPU; no paid compute and no process
  intervention.
