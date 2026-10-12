# Fixed VIDIMU sensor arm: completed, fusion comparison still pending

Source `9bdb68a`, command `python -m scripts.train_vidimu_sensor_arm`.
Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/vidimu-sensor-arm-01`.
CPU float32, Torch2.6.0+cu124, seed17, full batch130,100epochs, AdamW0.001/wd0.01.
4,719 trainable linear-head parameters. Frozen eight-bin orientation features
and actual sensor-placement metadata; no video/text encoder or audit read.
100updates/13,000 sample encounters,130 unique trials from10 subjects.

The earliest minimum validation NLL selected epoch100. Best and final weights
are saved separately; exact best reload probability parity passed. Best SHA256:
`b7c76c036fa3e66b4a71876c08772396d75f1421f31a01e15eac69747a8715d2`.
Fit/evaluation/artifact loop took1.274seconds, excluding process startup and
input loading. GPU training time and cloud cost are zero. Peak process RAM was
not measured in this runner and remains unreported rather than inferred.

| Epoch | Train Accuracy | Train NLL | Validation Accuracy | Validation NLL |
|---|---:|---:|---:|---:|
| 0 | .10769 | 2.73302 | .10256 | 2.79329 |
| 25 | .93077 | .51729 | .92308 | .53816 |
| 50 | .97692 | .26759 | 1.00000 | .27720 |
| 75 | .98462 | .18325 | 1.00000 | .20195 |
| 100 | 1.00000 | .13727 | 1.00000 | .16370 |

Selected development validation:39/39 correct (three held-out subjects),
NLL0.163704, Brier0.065464, ECE10bins0.135254. Placement-only Laplace prior:
Accuracy0.153846, NLL2.032392, Brier0.849735, ECE0.016050. The small ECE of that
weak prior is not good decision quality. All101 epoch predictions, IDs, targets,
normalization and histories are retained; per-class/subject/placement metrics
are in result.json. No observed NLL reversal at the tabulated points; this is
not proof against overfitting or a clean plateau, and no extra epochs are added.

This is a narrow activity task, only39 development trials after checkpoint
selection. It does not demonstrate general sensor understanding, synchronized
fusion, or95% quality across text/image/audio/video. Video can no longer improve
this arm's observed validation Accuracy; fusion must still be compared honestly
on NLL/Brier and perturbation controls. Keep the fixed video/joint protocol
unchanged and retain negative outcomes. No product promotion or compression.

## JSON output replay

`classification_json` maps fixed-class probabilities to caller option IDs and
serializes deterministically; it requires the complete class set, rejects invalid
probabilities and uses semantic class ID for ties. Five CPU tests passed.
All39 saved selected-checkpoint probability vectors were serialized and compared
against reversed option order, with exact class probabilities and selections
preserved. Outputs: `C:/CodexArtifacts/pretrained-reuse-r2/vidimu-sensor-json-replay-01`.
This checks the output layer using saved predictions, not another model forward
pass or support for arbitrary semantic candidate descriptions. It adds no neural
parameters, and JSON values are not asserted to be perfectly calibrated.
