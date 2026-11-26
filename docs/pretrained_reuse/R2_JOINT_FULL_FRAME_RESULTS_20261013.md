# Full-frame spatial preprocessing: completed comparison

All99 train/257 observed development videos retained. Completed both fixed50
update fits, exact best reloads, prediction reaggregation and paired comparison.
Followed the [predeclared protocol](R2_JOINT_FULL_FRAME_PROTOCOL_20261013.md).
No sealed evaluation, additional seeds, model adaptation or product promotion.

| Selected input | Best epoch | Accuracy | NLL | Brier | ECE | Both correct |
|---|---:|---:|---:|---:|---:|---:|
| Full frame |4|0.354086|1.087770|0.659664|0.048330|0.190661|
| Prior random crop |27|0.326848|1.096783|0.665425|0.041976|0.167315|
| Black frames |50|0.305447|1.103170|0.669794|0.049196|0.167315|

Full-frame minus crop paired video-bootstrap95 intervals: Accuracy
[-0.021401,0.075875], NLL[-0.020886,0.003315], Brier[-0.013916,0.002700].
All include zero. Draws10,000, seed17, video-cluster replacement/percentile95.
Checkpoint selection and prior development exposure limit interpretation.
The numerical improvement is not established as a reliable gain.

Changed-target pairs: full-frame1/127 both correct, crop0/127, black0/127.
Full-frame epoch0 Accuracy already0.354086; training does not improve it at the
selected point. Epoch0 validation NLL1.087863, selected1.087770. By epoch50,
train NLL falls to1.097864 while validation NLL rises to1.093394 and Accuracy
falls to0.346304. Best-selection correctly retains epoch4 rather than the end.
This is an overfitting warning; it does not establish a model-family ceiling.

| Epoch | Train NLL | Validation NLL | Validation Accuracy |
|---:|---:|---:|---:|
|0|1.118385|1.087863|0.354086|
|4|1.115380|1.087770|0.354086|
|10|1.111360|1.087907|0.352140|
|20|1.106026|1.088800|0.348249|
|30|1.102288|1.090279|0.348249|
|40|1.099740|1.091932|0.352140|
|50|1.097864|1.093394|0.346304|

## Integrity and cost

All1424 normal/black question records across train/validation have identical
video/question/target/frame identities to the crop caches. All712 black-control
records match exactly, including features and probabilities. The learned black
checkpoint predictions also match exactly. Only normal spatial preprocessing
changed: direct224x224 resize retains the whole frame and distorts aspect ratio.

Train extraction614.973seconds; validation1403.117seconds, CPU FP32/four threads.
Runs overlap: summed loop time2018.090seconds is not elapsed makespan.
Normal readout0.393seconds; black0.355seconds, excluding load/preflight.
Train198 unique questions/99 assets; validation514 questions/257 assets.
Both fits consume9900 example presentations; selected normal epoch4 consumes792.
192 trainable parameters. Same11,430,720-parameter scorer accounting as crop run;
this is not the full Omni product. No cloud charges or GPU training.

Artifacts:
- C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-tiny-fullframe-train-features-01
- C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-tiny-fullframe-valid-features-01
- C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-joint-fullframe-readout-01
- C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-joint-fullframe-report-01.json

Train feature SHA256329bb4e3541d4458239613f3f13e8dc7bd415dd13370256db65a6316508aa94d.
Validation SHA2568bb5c48cb18e66d9fa7231de19ec755e3e9fc35b72f22dcdea48056fe6af7ebb.
Runner source0419487; config records corpus/checkpoint/protocol hashes, and
prediction files record code hash/environment. Histories contain epoch0–50.
Best/final weights are separate with hashes; old outputs remain unchanged.

```powershell
python -m scripts.train_joint_video_readout --train C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-tiny-fullframe-train-features-01 --valid C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-tiny-fullframe-valid-features-01 --checkpoint C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-tiny-weight-audit-01/all-in-one-tiny.ckpt --output C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-joint-fullframe-readout-01
python -m scripts.report_joint_video_readout --run C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-joint-fullframe-readout-01 --reference-run C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-joint-readout-01 --output C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-joint-fullframe-report-01.json
```

## Decision

Do not adopt the model or expand this crop/resize sweep. Full-frame retention
addresses a real input-information risk but does not supply the missing temporal
object identity and condition-following behavior. Further work needs a distinct
evidence-preserving temporal representation hypothesis, not more tuning of this
same observed-validation readout. Broad95–98% Omni quality remains unachieved.
