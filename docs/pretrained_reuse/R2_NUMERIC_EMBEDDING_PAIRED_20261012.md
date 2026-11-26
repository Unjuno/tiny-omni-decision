# Paired numeric-delta development diagnostics

Saved predictions only; no model inference, training, new data or checkpoint selection.
All0/128/256/384/512 pairs match IDs, targets and option order. Resample underlying
state IDs with all questions together, then divide summed differences by sampled
question counts. Original group sizes differ, so equal-state averaging would not
match the reported sample-weighted Accuracy.10000draws,seed17,percentile95%.

At512 original text (2540 questions/732 states):11 newly correct,3 newly wrong,
net8 (+0.315pp; interval[+0.039,+0.604]pp). NLL worsens+0.001524
([+0.000491,+0.002606]); Brier worsens+0.000555. Accuracy and probability quality
therefore move in different directions.

At512 curriculum (2048 questions/256 states):20 newly correct,19 newly wrong,
net1 (+0.049pp; interval[-0.586,+0.684]pp). NLL improves−0.006859
([-0.009694,-0.003862]); Brier improves−0.003286. Numeric-row updates have a
small measurable effect, not a demonstrated Accuracy improvement or95% result.

These are post-hoc descriptive intervals on repeatedly observed development data,
multiple steps/metrics without multiplicity correction. Do not reinterpret them
as blind significance tests. Both runs still select identical step0 checkpoints.

Next comparison: a separate fresh2048-step run is justified by the parent learning
later than512 and this mixed early response. Freeze all settings, require exact
512-prefix reproduction, and stop at2048. No optimizer-resume claim, extra seed,
LR search, new transfer evaluation, audit or adoption from these diagnostics.
Artifact: `C:/CodexArtifacts/pretrained-reuse-r2/numeric-embedding-paired-01`.
