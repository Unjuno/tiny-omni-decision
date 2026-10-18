# Full-frame preprocessing comparison

Completed: [results](R2_JOINT_FULL_FRAME_RESULTS_20261013.md). All extraction,
fixed fits and paired comparisons finished. Quality remains insufficient.

The preceding train annotation diagnostic found the random crop fully excluded
the marked occluder at frame0 in24/98 annotated training clips. Test full-frame
retention without using any box annotation to construct model inputs.

Change only the spatial transform from Resize256/RandomResizedCrop224 to direct
Resize((224,224)), followed by the same tensor conversion and normalization0.5.
This retains the whole image but distorts non-square aspect ratios. It is a
predeclared trade-off, not a claim of generally optimal preprocessing.

Keep the exact three pre-cut frame IDs, seed17 patch sampling, text serialization,
base checkpoint, frozen features, train99/validation257 videos, and all questions.
Reuse the fixed192-weight, pretrained-ITM initialization, full-batch AdamW0.001,
weight_decay0.01,50 updates, minimum-validation-NLL selector including epoch0.
Do not use the separate train-only LBFGS diagnostic weights or schedule.

Normal and black-frame features remain paired. Black input should exactly match
the previous feature cache because it is always the same normalized224x224 black
tensor; verify this before accepting the new cache. Save separate feature/run
directories and hash every completed prediction cache. Keep the crop baseline.

Report selected and final metrics, full learning curves, changed-target pair
performance, and paired video-bootstrap differences against the fixed crop
baseline (10,000 draws/seed17). An improvement in NLL alone does not establish
temporal reasoning. Do not select a different transform, seed or budget from
these results. This remains observed development validation, never a sealed audit.

CPU FP32/four threads, initial one-training-video smoke, then full frozen feature
extraction if finite outputs and unchanged black-control features are confirmed.
No encoder updates, cloud compute, product promotion or old artifact replacement.

## Training cache admission completed

Full-frame train99 videos /198 questions completed in614.973seconds CPU.
Normal/black feature shapes both[198,3,192];396 rows total. All198 black rows
(including every feature and probability) exactly match the crop cache; all396
row identities, targets and frame IDs match. No accuracy-based selection made.
New cache SHA256329bb4e3541d4458239613f3f13e8dc7bd415dd13370256db65a6316508aa94d.
Artifact: all-in-one-tiny-fullframe-train-features-01/admission.json.
Validation extraction is still running; no full-frame readout fit yet.
