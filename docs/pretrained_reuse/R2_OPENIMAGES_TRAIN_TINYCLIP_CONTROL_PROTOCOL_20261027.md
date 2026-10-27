# Fixed TinyCLIP control on train candidate images

Freeze before model inference. Use all 257 rows of the already frozen
four-choice train-candidate task and corresponding saved JPEGs; require unique
IDs, exact per-image SHA-256 and matching earlier Flickr page declarations.
Pin the candidate task, media manifest, page receipts, scope and the existing
TinyCLIP checkpoint; never access validation or test labels. Run the already
audited 83,101,185-parameter TinyCLIP on CPU, eval mode, with the prior prompt
`a photo containing {label}.`, fixed revision and no calibration or tuning.

Report Accuracy, NLL, Brier and 15-bin ECE on the same 257 examples for aligned
images, a fixed seed-20261026 image derangement, and a uniform image filled
with the processor image mean. Compute a single fixed set of 11-label logits
per image and reuse them for the derangement; score the uniform image once.
Report their top-1 agreement and correctness changes relative to aligned, and
the prior option-only cross-fold diagnostic on exactly these tasks as a separate
reference. Save private per-example probabilities and source hashes; no model
selection, weight changes, prompt search or extra training.

This is a train-candidate development diagnostic after task priors were observed;
the model was pretrained on YFCC15M and source-level pretraining overlap is not
excluded. It is not blind validation, and cannot establish general image or
multimodal quality or rights ownership. These task rows are not yet admitted for
training. No comparison to a nonidentical set may be treated as an improvement.
