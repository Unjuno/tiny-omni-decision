# Frozen DINOv2 image candidate baseline

Goal: test another already-audited small pretrained representation under the
existing CLEVR-4 task/coverage, before considering new backbone training.
This is synthetic taxonomy choice quality, not general conditional image reasoning.
Do not equate a smaller encoder with success, or call this a video model.

facebook/dinov2-small revisioned25f3a31f01632728cabb09d1542f84ab7b0056;
Apache2 checkpoint SHAae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1.
22,056,576 frozen parameters. Use FP32/eval and concatenate CLS384 with mean
of256patch tokens384, giving768features, same dimension as prior VideoPrism.
Images are640x480. To avoid dropping objects counted in labels, explicitly resize
the whole image to224x224 bicubic without crop, then use pinned processor channel
normalization. This preserves contents but distorts aspect ratio; do not pretend
it is the previous encoder's preprocessing or an encoder-only causal comparison.
No alternate pooling/preprocessing selected after metrics.

Same8,424train IDs/order and512already-observed selection validation IDs/order;
do not open the remaining1,595reporting-validation or sealed audit data.
Train-label SHA00f03f27982e2785636e85a50f4ecf258c0ac3ee483d7242e297cee14b50ec2e;
train-image-manifest SHA2c56815beb9d2e9951cd38686eafdbe2ecdee05e69e74b5acfad24fbefbb9010.
Validation-label SHA3c213f13ef2214baf1130cbe49c1d981b0125bbff3813e4ac5b4e341e443d4a1;
validation-image-manifest SHA77faf433844be6b350cd0205fd3834cd8b028a6491ddd9eae0f6e9b7610cd38a.
Verify unique disjoint IDs/media hashes and every image hash. Keep10options for
color/texture/count/shape and all examples. No label simplification.

Same frozen MiniLM-L3 candidate embeddings and shared CandidateScorer
(1152→128→1,147,713parameters), seed17, AdamW1e-3,weight_decay1e-4,
batch64questions,8epochs. Earliest minimum validation NLL; no extensions/seeds.
Save best/final independently; verify saved-best reload. Train encoders never.
Report all epoch metrics, per-taxonomy Accuracy/NLL/Brier/ECE, unique exposure,
training versus validation gaps, total parameters, time/VRAM and artifact hashes.
Compare saved prior VideoPrism512-selection predictions, retaining its historical
results. Different feature/preprocessing means no single-factor causal claim.

Separate16-image/one-epoch smoke first, including encoder repeat and finite checks.
Full outputs new dinov2-clevr4-probe-01, no old artifacts modified. At selected
checkpoint additionally score fixed cyclic+1image features with unchanged
questions/labels to test visual dependence; retain coincident taxonomy matches.
No final audit or compression. Local GPU only, no cloud costs. If weak, keep the
failure rather than repeat arbitrary seeds or narrow the task.
