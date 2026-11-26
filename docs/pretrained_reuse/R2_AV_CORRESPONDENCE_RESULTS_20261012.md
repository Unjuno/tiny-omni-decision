# Fixed audio/video correspondence development result

## Execution

Source `fa9fe9d`; protocol fixed before fitting. All456 windows and audio/video
features passed hashes, exact IDs and repeat-extraction checks. Audio extraction
took189.289 seconds; video1123.286 seconds (overlapping CPU execution, do not sum
as wall time). Both repeat maxima were zero. No encoders were trained.

`python -m scripts.train_av_correspondence --windows <windows>/result.json
--audio <audio> --video <video> --output <training>` completed100 epochs,
6400 updates,25600 presentations of256 unique train recordings. Validation has
200 recordings /50 quartets. Seed17,64D bias-free projections,73,728 trainable
parameters, temperature0.1,AdamW0.001,weight decay0.01. Fit/history took10.088
seconds on CPU. Earliest minimum validation NLL selected epoch1 (64 updates,
256 presentations). The final checkpoint remains separate. Both reloads exactly
reproduced recorded metrics; candidate reversal passed logit tolerance1e-6.

All directories below are under `C:/CodexArtifacts/pretrained-reuse-r2/`:

- Windows: `perception-correspondence-development-windows-01`.
- Audio/video: `perception-correspondence-development-audio-01` and
  `perception-correspondence-development-video-01`.
- Fit: `perception-correspondence-development-training-01` (config/input hashes,
  100-epoch history, all control predictions, best/final weights, executed sources,
  bootstrap results).

Best296326 bytes, SHA256
`01c43f9ad0f9cc3486535560de872ac931809cce619658372ac15ab421891c69`.
Final296332 bytes, SHA256
`aa38895c604c9ef11bc09ca6e25b7830b71fcc11bfc93b1ba2d9ce2400d10594`.
Prediction SHA256
`e8fca0e277f1ecb12c8cbc81a8bb6a570f5f8c9e78f32ab8038235a6f8b1376a`.
Weights/media remain outside Git. Existing artifacts are unchanged; no paid
compute, GPU training, sealed audit or product promotion occurred.

## Same-validation comparison

| Condition | Accuracy | NLL | Brier | ECE15 |
|---|---:|---:|---:|---:|
| Untrained projection |0.250|1.398207|0.756641|0.055991|
| Selected epoch1 |0.510|1.157063|0.629225|0.070286|
| Final epoch100 |0.530|1.473391|0.660546|0.239584|
| Selected, audio zero |0.250|1.386294|0.750000|0.000000|
| Selected, video zero |0.250|1.386294|0.750000|0.000000|
| Selected, cyclic query audio shift |0.200|1.708910|0.887925|0.239714|
| Selected, reversed options/remapped targets |0.510|1.157063|0.629225|0.070286|

Zero controls are uniform by construction, not independent evidence of semantic
complementarity. The shifted audio control breaks recording correspondence.

## Learning curve

| Epoch | Train Accuracy | Train NLL | Validation Accuracy | Validation NLL | Validation ECE |
|---|---:|---:|---:|---:|---:|
|1|0.886719|0.655949|0.510|1.157063|0.070286|
|5|0.992188|0.048642|0.570|1.247057|0.166038|
|10|1.000000|0.003014|0.590|1.319954|0.204042|
|25|1.000000|0.001216|0.580|1.345317|0.199076|
|50|1.000000|0.000480|0.565|1.390048|0.226220|
|100|1.000000|0.000116|0.530|1.473391|0.239584|

Training memorization accompanies worsening validation NLL/calibration. Keep the
predeclared NLL selector; do not switch to Accuracy to select a later checkpoint.
This is overfitting evidence, not justification to add epochs/capacity.

## Uncertainty and decision

10,000 shared quartet-cluster bootstrap draws, seed17,95% percentile intervals:
selected Accuracy[0.435,0.585], NLL[1.090296,1.224808],
Brier[0.592372,0.666525], ECE[0.048527,0.156093]. Paired selected-minus-untrained
Accuracy[0.180,0.335], NLL[-0.303769,-0.177084],
Brier[-0.162645,-0.092524], ECE[-0.019088,0.106225].
Checkpoint selection used this development validation, so these are not blind
quality guarantees. Quartets are the minimum dependency unit; participant,
script/environment overlap and acoustically ambiguous negatives remain limits.

Frozen features contain learnable recording-alignment information in this small
development sample. This does not demonstrate general multimodal decision
reasoning or95–98% modality quality. Close this fixed comparison without more
validation-driven fitting. Retain the selected readout only as a research probe;
the next product gate remains condition-sensitive decisions on a prospective,
task-matched battery with stronger generalization checks. This result alone does
not justify shared-Lens adaptation, weight merging or quantization.
