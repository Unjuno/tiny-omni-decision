# Fixed CAV-MAE Sync correspondence evaluation

Executed `python -m scripts.evaluate_cav_sync_correspondence` on the existing
200 development-validation recordings/4-option routes. No fitting, calibration,
prompt search, checkpoint selection or audit-set access. Official weights remain
REVIEW for product use; this is a research comparison.

| Metric | Frozen CAV-MAE Sync | Uniform probabilities |
| --- | ---: | ---: |
| Accuracy | .3200 | .2500 (balanced target positions) |
| NLL | 1.370076 | 1.386294 |
| Brier | .741869 | .750000 |
| ECE (10 bins) | .052695 | .000000 |

Scoring is cosine similarity between audio/image CLS vectors followed by softmax
at fixed temperature1. All candidates retain the pre-existing order and target;
the routing hash is
`03d389b0eda847b4d16e0205a7c285ad78ebecfb6405fde1c85c0ddfd569d395`.
No target IDs enter model inputs. All200 features passed finite checks and exact
repeat equality. No examples were dropped. GPU inference only; no model updates.

Preprocessing is the fixed real-input smoke recipe: retained middle frame and
PTS-centered4.175s audio,416 Kaldi frames, upstream normalization. It is one
frame and local audio, **not** upstream16-frame retrieval. Legacy288-square
frame resizing remains a limitation. Different preprocessing and no trained
projection mean this is not an isolated backbone comparison against the earlier
trained correspondence model. Neither task proves semantic multimodal decisions.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r1/cav-mae-sync-correspondence-valid-01/`:
`features.npz`, `predictions.npz`, and `result.json` retain features, probabilities,
targets, logits, ordered recording IDs, time/sample intervals and source hashes.
Feature extraction timing and peak allocated CUDA memory are in result.json.

Conclusion: only a small improvement over uniform NLL/Brier and32% accuracy.
Do not promote this configuration, tune its temperature on these results, or
claim the model family cannot work. The published multi-frame aligned readout
has not been evaluated here. Existing artifacts and model weights are unchanged.
