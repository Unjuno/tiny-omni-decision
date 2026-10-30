# Fixed training-fit diagnosis

Source18d849f. Inference only, no selection, new output directory. Seed83 selects
32states per numeric transform,128states/1,024questions in total. Every selected
question ID was consumed before each model's selected checkpoint. Same latent
state IDs, each model evaluated on its own training values: narrow original and
wide numeric-range training respectively. This is not a same-input causal
comparison. Sampling uses only group IDs and transform strata, not correctness.
Both exact checkpoints and corpora are pinned by SHA; all inputs fit512tokens.

| Model / own training slice | Accuracy | NLL | Brier | ECE15 |
| --- | ---: | ---: | ---: | ---: |
| narrow | 0.82324 | 0.42472 | 0.23660 | 0.02858 |
| numeric_range | 0.27246 | 1.43165 | 0.75845 | 0.02059 |

The wide-range model is near chance even on consumed training questions. This
supports inadequate fitting of this curriculum within the measured protocol;
poor transfer alone is not the explanation. The narrow model's sampled training
Accuracy82.32% and full in-distribution validation80.57% provide a useful contrast,
but the sets have different states and no confidence-bound gap is asserted.
The wide model's training27.25%, narrow-format validation24.51% and new-range
transfer24.27% are different distributions, not a directly estimated same-domain
generalization gap. No evidence here establishes a universal capacity ceiling,
proves overfitting absent, or separates optimization from numeric representation.

Do not promote the wide-range recipe or infer usefulness from its low ECE.
Any next experiment should address training fit using a predefined budget and
validation-only selection before exposing new transfer outcomes. Preserve the
stronger narrow-format checkpoint and all failures. No extra training was run
in this diagnosis; no sealed audit or product quality claim.

Local inference 8.986s, peak PyTorch allocation 193,215,488bytes. No cloud cost. Exact paths, hashes, per-task metrics and all predictions are retained in adjacent JSON and external artifacts.
The execution commit had a line-length-only Ruff issue; formatting was corrected
after execution without rerunning or changing the measured inference.
