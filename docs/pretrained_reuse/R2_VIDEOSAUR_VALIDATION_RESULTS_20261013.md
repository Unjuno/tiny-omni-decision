# Fixed VideoSAUR endpoint: development validation

All257 validation clips completed extraction successfully. The fixed epoch50
head was evaluated on all514 questions, with no checkpoint selection, additional
training or calibration. Source commit: f67d491. Frozen checkpoint hashes and
input hashes are preserved in the accompanying JSON and local evaluation ledger.

| Input / head | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Normal / trained | .361868 | 2.733249 | 1.000905 | .436157 |
| Mismatched video / trained | .354086 | 3.212561 | 1.016037 | .438792 |
| Zero video / trained | .258755 | 1.105744 | .671483 | .094312 |
| Normal / untrained | .305447 | 1.099097 | .666991 | .031792 |

Train Accuracy was .757576 and NLL .443811: the fixed endpoint exhibits a large
generalization gap and overconfidence. Initial-question Accuracy is .392996,
final-question Accuracy .330739. Both questions are correct on45/257 videos;
among127 changed-target videos, only3 have both answers correct. There are20
prediction switches between conditions. This does not meet the Decision gate.

Normal-minus-mismatch paired video bootstrap (10,000 replicates, seed17,
95% percentile intervals) gives Accuracy difference .007782
[-.052529,.068093]. NLL difference -.484306 [-1.003404,.016410].
Both intervals include zero. The bootstrap uses a declared probability floor
of1e-15; its NLL difference therefore differs slightly from the existing metric
implementation's reported NLL difference. These are previously observed
development data, not blind evaluation or independent confirmation after search.

The train-only improvement did not establish useful validation transfer. Preserve
this negative result; do not promote the candidate, select another epoch from
this validation, or interpret it as a universal limit of object-centric models.
The current frozen representation plus small candidate-attention head does not
yet demonstrate reliable condition-dependent temporal reasoning.

## Reproduction and artifacts

```powershell
python -m scripts.evaluate_slot_readout --features C:/CodexArtifacts/pretrained-reuse-r1/videosaur-valid-slots-01 --output C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-slot-valid-evaluation-01
```

The output directory already exists and is protected against overwrite. It holds
examples.json, predictions.json (both heads and all three controls), and result.json.
The evaluation completed on CPU in .7775 seconds excluding admission/loading.
Saved checkpoint hashes remained unchanged. The original train extraction,
training artifacts and all legacy runs were preserved. No sealed audit was used.

Independent NumPy recomputation from saved predictions matched all four aggregate
metrics for both heads and all three controls within1e-12. Prediction/example file
hashes matched the saved ledger. See R2_VIDEOSAUR_VALIDATION_RECHECK_20261013.json.
Extraction processed122,973 frames in11,902.759 seconds (3.306 hours); peak CUDA
allocated memory was705,276,928 bytes and reserved memory893,386,752 bytes.

Remaining work: assess the next R2 action against the roadmap. The complete Omni
objective remains unmet. Neither shared-Lens nor merging is justified by this result.
