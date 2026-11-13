# Fixed context encoder-coverage comparison

Motivation: consumed training examples remained weak after final-block adaptation.
One new intervention changes trainable scope to all six encoder blocks/classifier.
Same parent, same3,072 ordered training questions/views, same2,304development rows,
seed17,384updates,AdamW2e-5/.01,clip1,dropout disabled. A separate two-update smoke
passed before launch. No intermediate checkpoint selection or budget extension.
Initial per-example predictions are byte-identical to the final-block experiment.

| Context | Final-block Accuracy | All-block Accuracy |
|---|---:|---:|
| Plain | .915365 | .937500 |
| Irrelevant prefix | .535156 | .932292 |
| Necessary prefix | .895833 | .936198 |

All-block scheduled accuracies at128/256/384 are respectively:
plain .927083/.932292/.937500; irrelevant .917969/.932292/.932292;
necessary .925781/.937500/.936198. Full NLL/Brier/ECE and exact measurements
are in the adjacent JSON. All scheduled predictions and final weights retained.
This supports broader adaptation as useful under this fixed corpus/budget, not
a unique proof of representational capacity as the cause. No additional runtime
parameters were introduced; more existing pretrained weights were updated.

Trainable scope10,647,169parameters versus1,774,849. Embeddings/pooler remain frozen;
frozen hash and exact final reload pass. Final candidate reversal logits match at
atol/rtol1e-5 over all development rows. Same initial data/config/order fields were
independently checked after completion. Learned encoder delta is saved separately
as final-encoder.pt, preserving the parent and last-block runs.

This is a promising research candidate for context robustness, not product adoption.
Known comparison words, synthetic states and held-out context templates do not
establish unseen language/domain transfer, original-task retention, or multimodal
quality. No old transfer benchmark or sealed data was reopened. Next evidence must
measure those remaining gaps under a frozen protocol, without selecting a lucky
epoch or replacing the final checkpoint using observed scores.

Run: `python -m scripts.train_context_robustness --arm context --policy all_encoder_blocks`.
Source5f64435. Local outputs: C:/CodexArtifacts/pretrained-reuse-r2/
context-robustness-context-run-01-all-encoder (and separate smoke suffix).
No cloud use. Resource totals and artifact hashes retained in result.json.
