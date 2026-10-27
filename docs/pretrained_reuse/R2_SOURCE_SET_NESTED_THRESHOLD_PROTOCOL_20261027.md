# Nested source set-decision threshold experiment

Goal: test whether fixed 0.5 decisions underlie the observed set-recall and
none-visible tradeoff. This is a nonblind exploratory follow-up on the same
train-source videos, not a sealed or independent evaluation.

For each existing outer video-group fold, exclude all outer videos, hash the
remaining videos using `set-inner-v1:17:<video_id>`, sort by digest and ID,
and alternate into two group-disjoint inner folds. Train one joint CandidateModel
per inner split (six fits total), seed 17, seven epochs, AdamW 0.001, weight
decay 0.0001, batch 32 and clip 5; frozen features and model match prior work.
Save inner checkpoints, per-record conditional/none probabilities and hashes.
Select thresholds only using pooled inner held-out predictions of that outer
training set. Grid membership {0.2,0.35,0.5,0.65,0.8}, none {0.35,0.5,0.65};
maximize unweighted macro exact accuracy over none, one and multi-target strata.
Break ties by smallest distance from (0.5,0.5), then lower membership and none
threshold. Decode visible membership with fallback all maximum-score ties
within 1e-6 of the maximum logit; none is exclusive. No calibration is claimed.

Apply each frozen pair to the existing held-out outer joint probabilities;
compare against the original 0.5/0.5 decisions on the same examples, overall
and by stratum, with paired 320-video percentile bootstrap (10,000 draws,
seed 17). Do not train on or tune using outer records while selecting their
threshold. OOF models share training data and contributors may cross videos;
the threshold transfer to models trained on more data and repeat-development
reuse limit causal and generalization claims. No sealed audit, product selection
or repeated threshold runs. Preserve every existing artifact; stop if disk falls
below 1 GiB, nonfinite values occur or provenance/group checks fail.
