# Exact training-frame crop retention diagnostic

The prior joint representation experiments used the released pixelbert transform:
Resize256 then RandomResizedCrop224 (default area0.08–1, aspect0.75–1.333), seed17
reset before each video's three independently transformed frames. This is a
specific audit preprocessing choice, not proof of task-appropriate preprocessing.

Replayed crop parameter sampling using actual decoded first-frame dimensions,
the same resize and RNG sequence, and the immutable training annotations. Only
exact frame0, unmasked, marked-occluder boxes were compared; no interpolation,
validation metrics, model inference or new training.

| Exact frame0 evidence | Videos |
|---|---:|
| Total training videos |99|
| Frame0 marked-occluder box available |98|
| Box completely outside first crop |24|
| Box center outside first crop |29|
| Less than half box area retained |29|
| Entire box area retained |51|

One missing frame0 annotation remains explicitly unassessed. These counts overlap
and must not be summed. They describe geometry, not per-example model correctness.
Later frames and other contextual cues may still provide useful information.
Nevertheless, dropping the annotated occluder from24/98 initial views is a concrete
input-information risk and weakens any inference of a representation-family ceiling.

The next admissible comparison should preserve the full frame, holding frame IDs,
question text, model weights and the predefined readout optimization fixed. It must
retain all questions and use no annotation-guided cropping at inference. Do not
overwrite previous caches or reinterpret their scores as full-frame results.
Any resize distortion/padding choice must be specified before scoring.

Artifact: C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-joint-crop-diagnostic-01.json.
Executed script: C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-source-initial-01/crop_diagnostic.py.
Script SHA256598b9624f3eda8241e599f4e79de58d921e268e12d18a11db7773549666e14d8.
Result SHA256140b76695dcb810116c22d47e89a24ee7f977831976b14d66e7c609aac7d09c7.
Annotation SHA2569eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78.
Media manifest SHA256469299ad240ff85c47ba44abb8920aae9994b0e3ec15a3fca1373c49b2fa6ee3.
No GPU, paid compute, sealed audit, artifact replacement or product promotion.
