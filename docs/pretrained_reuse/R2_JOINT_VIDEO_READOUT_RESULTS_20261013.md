# Fixed joint-feature Decision readout results

Completed the [fixed protocol](R2_JOINT_VIDEO_READOUT_PROTOCOL_20261013.md).
Do not adopt this readout. All 99 train videos /198 questions and257 observed
development videos /514 questions were retained. No sealed audit, additional
seed, calibration or backbone optimization. Both models ran50 full-batch updates.

## Selected checkpoints

| Input | Selected epoch | Validation Accuracy | NLL | Brier | ECE | Both conditions correct |
|---|---:|---:|---:|---:|---:|---:|
| Normal video |27|0.326848|1.096783|0.665425|0.041976|0.167315|
| Black frames |50|0.305447|1.103170|0.669794|0.049196|0.167315|

Both-correct among127 videos whose target changes: **0/127 for both models**.
Normal initial/final condition Accuracy:0.291829/0.361868. Black:0.291829/0.319066.
Normal's untrained initial validation Accuracy was0.332685 and NLL1.099540;
training improves NLL slightly while Accuracy falls. This is not a useful
condition-sensitive temporal Decision model.

Paired video-cluster bootstrap (10,000 draws, seed17, percentile95), normal minus
black: Accuracy[-0.009728,0.054475], NLL[-0.012083,-0.000885],
Brier[-0.008273,-0.000629], both-correct[-0.027237,0.027237]. These intervals
describe the observed validation after checkpoint selection, not blind performance.
Black images are an out-of-distribution control; lower NLL alone is insufficient
evidence of correct scene-specific temporal reasoning.

## Learning curves

| Epoch | Normal train NLL | Normal valid NLL | Normal valid Accuracy | Black train NLL | Black valid NLL |
|---:|---:|---:|---:|---:|---:|
|0|1.123952|1.099540|0.332685|1.213399|1.194013|
|10|1.116988|1.097820|0.330739|1.180735|1.164322|
|20|1.111056|1.096939|0.326848|1.153839|1.140695|
|27|1.107543|1.096783|0.326848|1.138533|1.127778|
|30|1.106194|1.096818|0.326848|1.132831|1.123109|
|40|1.102336|1.097300|0.326848|1.117246|1.110936|
|50|1.099343|1.098190|0.324903|1.106228|1.103170|

Normal training NLL continues falling after epoch27 while validation NLL rises
slightly. This is an overfitting warning, not proof of a capacity ceiling.
Selected normal train Accuracy0.262626; final0.308081. Black Accuracy remains
0.252525 train /0.305447 validation. Full51-point histories are retained.
Do not change the selector to Accuracy or extend epochs based on these results.

## Accounting and artifacts

- Frozen matching backbone11,430,914 parameters includes its386-parameter ITM head.
  Replacing that head with192 weights gives11,430,720 resident parameters for
  this video/text scorer. This is not a full Omni product parameter inventory.
- Each readout trains192 parameters, uses198 unique questions/99 videos,
  and9900 example presentations over50 updates. Selected normal epoch27
  represents5346 presentations; selected black50 represents9900.
- CPU FP32/four Torch threads, Python3.11.9/Torch2.6.0+cu124; no cloud/GPU training.
  Train feature extraction600.539s; valid1383.610s. The sum is1984.148s of
  extraction-loop wall durations; runs overlapped, so this is not elapsed makespan.
- Readout fit/selection/reload wall time: normal0.472s, black0.356s.
  Excludes feature loading/preflight. Both exact best-reload checks pass.
- Sampled validation extractor working set779,825,152bytes; process peak up to
  that sample847,773,696bytes. Full-run peak and deployment latency not measured.
- Each best readout file1869bytes; final1874bytes. Full best/final hashes,
  predictions and histories are preserved. No legacy artifact overwritten.
- Training command source commit4f11247. The initial direct-file invocation
  failed import before creating an output; use module execution below.
- Graph rendering was not run: matplotlib is absent. The complete numeric
  learning curves above and in history.json remain available; no environment upgrade.

Output: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-joint-readout-01/`.
Report: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-joint-readout-report-01.json`.
Train feature SHA256:`8d061543600446f0dfd659feeb1e88c6c039c7f8cfab3104141c92a5fe9be3ef`.
Valid feature SHA256:`dd85d8cb4a70bd18b08817c3591fcedf1e982e16a62ebe602427eb1288f32b42`.
Config includes corpus/checkpoint/protocol hashes; prediction files include source
commit, runner hash and environment. Model weights licensing remains REVIEW.

```powershell
python -m scripts.train_joint_video_readout --train C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-tiny-train-features-01 --valid C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-tiny-valid-features-02 --checkpoint C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-tiny-weight-audit-01/all-in-one-tiny.ckpt --output C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-joint-readout-01
python -m scripts.report_joint_video_readout --run C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-joint-readout-01 --output C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-joint-readout-report-01.json
```

Existing outputs are protected; these commands refuse overwriting them.
The broad multimodal quality/fusion gates remain unmet. This bounded result
closes the specified linear-readout experiment, not the model family or goal.
