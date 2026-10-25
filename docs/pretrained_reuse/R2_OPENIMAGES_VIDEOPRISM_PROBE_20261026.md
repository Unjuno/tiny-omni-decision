# R2 Open Images V6 frozen VideoPrism probe — 2026-10-26

## Result

A frozen VideoPrism-B image representation plus frozen MiniLM candidate-text
features and a 147,713-parameter scorer reached **79.17% Accuracy** on the
fixed 168-example development validation set (11 hierarchy-selected classes,
four choices each). Selected validation metrics were NLL **0.5609**, Brier
**0.3014**, ECE-15 **0.1454**, and mean confidence **0.6931**. The uniform
four-choice baseline was 21.43% Accuracy, NLL 1.3863, Brier 0.7500. The
training-position majority baseline was 30.36% Accuracy. Macro Accuracy over
the 11 target classes was 78.41%.

This is a narrow image-only development result. Validation selected the
checkpoint and is not a blind test. It does not establish general image
understanding, four-modality performance, or the 95–98% product target.

**Follow-up input control changes the interpretation:** the same selected
checkpoint gave exactly the same top-1 predictions on all 168 validation rows
when image features were zeroed or deranged across validation images. Accuracy
remained 79.17% in all three conditions. The aligned image slightly changed
confidence metrics, but did not change any choice. Do not interpret the result
above as evidence that the choices were image-grounded. The scorer's decision
ranking may come from text/candidate regularities; a separately trained
text-only scorer reached 75.60% with a smaller readout and fixed final epoch.
This is a diagnostic on the same exposed split, not a matched comparison. See
the [fixed input controls](R2_OPENIMAGES_INPUT_CONTROLS_20261026.md) and
[text-only control](R2_OPENIMAGES_TEXT_ONLY_CONTROL_20261026.md).

## Data gate and fixed splits

The task is Open Images V6 image-level object-presence selection using one
positive label and three explicitly negative hierarchy siblings. The question
is “Which object is present in the image?” Selected class scope and task rows
were frozen before this model run. The source split is train versus validation;
the pilot contains 168 available downloaded examples in each split. Failed
downloads were excluded without replacing tasks or changing sampling limits.

The split checks found zero overlap for ImageID, source OriginalMD5, normalized
RGB SHA-256, and perceptual hashes at Hamming distance <=4. The frozen sample
ID-order hashes are:

| Split | Examples | Sample ID order SHA-256 | Feature cache SHA-256 |
|---|---:|---|---|
| Train | 168 | `4f7bdf9c0f8d6adb911fb93e68e7d541d2f33786ccb937c6c1b11c42d0aca61e` | `a055656046ed7d83c4a3b3b3da9cdadaf4370a0671aafcc0b98df909e7ddfe46` |
| Validation | 168 | `785c3cd1b6fb260964b615fd27843063671a379baead722ae6a7b375ef8e1636` | `2b13b7b50e026aa2c8e975239784b1d4aac5ea6343a446ba0bf490ca15fe1622` |

All 336 media rows matched a current Flickr page license declaration and
attribution profile. `individual_rights_verified=false` remains recorded for
every row: page metadata is not independent proof of ownership or a product
rights clearance. The [Open Images V6 documentation](https://storage.googleapis.com/openimages/web/factsfigures_v6.html)
also says it does not verify the license of each image. Keep this pilot and its
learned readout in the research-only state.
Sealed audit data was not loaded.

Validation results by target class (small per-class counts are shown):

| Target | Count | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|
| Toy | 16 | 100.0% | 0.147 | 0.053 | 0.127 |
| Door | 14 | 64.3% | 1.074 | 0.610 | 0.282 |
| Food | 15 | 100.0% | 0.188 | 0.057 | 0.167 |
| Plant | 16 | 93.8% | 0.309 | 0.141 | 0.199 |
| Sculpture | 15 | 100.0% | 0.183 | 0.071 | 0.157 |
| Vehicle | 18 | 100.0% | 0.470 | 0.234 | 0.344 |
| Clothing | 16 | 56.3% | 0.800 | 0.445 | 0.329 |
| Furniture | 17 | 76.5% | 0.679 | 0.365 | 0.335 |
| Building | 13 | 38.5% | 1.144 | 0.668 | 0.374 |
| Window | 15 | 33.3% | 1.063 | 0.648 | 0.197 |
| Animal | 13 | 100.0% | 0.208 | 0.086 | 0.169 |

## Model and training

- VideoPrism-B: `google/videoprism-base-f16r288@be719a406d563b66f0ac969e7c94bab8e997c81a`, 114,365,184 parameters, checkpoint SHA-256 `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
- MiniLM-L3: `sentence-transformers/paraphrase-MiniLM-L3-v2@4ca70771034acceecb2e72475f72050fcdde4ddc`, 17,389,824 parameters, weight SHA-256 `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- Both encoders were frozen. The scorer receives the concatenated 768-D image and 384-D question-conditioned candidate-text features. It has 147,713 trainable parameters; total resident model count is **131,902,721**.
- Seed 17, AdamW, LR 0.001, weight decay 0.0001, batch size 64 decisions, eight epochs. The earliest minimum validation NLL was selected; epoch 8 was best. No temperature fitting or post-hoc calibration was done.
- Eight epochs showed train Accuracy 64.9%→73.2% and NLL 1.306→0.720; validation Accuracy 73.8%→79.2% and NLL 1.284→0.561. No overfitting signal appeared in this short curve, but validation reuse for selection and the small class counts limit inference.
- Best readout save/reload had maximum validation-logit delta 0.0. The final checkpoint is stored separately.

| Epoch | Train Acc. | Train NLL | Train Brier | Train ECE | Val Acc. | Val NLL | Val Brier | Val ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64.9% | 1.306 | 0.709 | 0.374 | 73.8% | 1.284 | 0.697 | 0.458 |
| 2 | 66.1% | 1.203 | 0.652 | 0.375 | 73.8% | 1.150 | 0.622 | 0.411 |
| 3 | 68.5% | 1.079 | 0.580 | 0.339 | 75.0% | 0.991 | 0.528 | 0.354 |
| 4 | 70.2% | 0.962 | 0.511 | 0.279 | 75.6% | 0.843 | 0.440 | 0.284 |
| 5 | 71.4% | 0.870 | 0.461 | 0.210 | 76.2% | 0.734 | 0.382 | 0.219 |
| 6 | 71.4% | 0.800 | 0.430 | 0.156 | 77.98% | 0.647 | 0.340 | 0.199 |
| 7 | 72.0% | 0.760 | 0.402 | 0.132 | 79.2% | 0.598 | 0.317 | 0.145 |
| 8 | 73.2% | 0.720 | 0.386 | 0.098 | 79.2% | 0.561 | 0.301 | 0.145 |
- Training config SHA-256: `3cca69b55316b2efeeb1d458025cf79a53dbc7a0e44d4c94183dba53d4ae225b`; feature config SHA-256: `d1496505f089a6e8e99dc4c8b4e549f265bac6455582f97d8ddf4361c12a6e5a`.
- The feature run recorded extractor SHA-256 `4eec701a8b94c50a6a3593a031bb14d641a139e22f3d05021bdabedc69706004` and worker SHA-256 `19ba73a1ff063966511b7eb04e423e9c9f53b9cb94d8a9a5a308c7319c3ce0ab`. The tracked extractor was subsequently formatted and its optional checkpoint-size metadata field was removed; current extractor SHA-256 is `6684281d6183a16ebd583ae42a19a3f37893e024c3f1104bb102ced2c4ac36d6`. Feature decode, preprocessing, model, split, and extracted outputs were not changed after the run. The pre-format extractor source was not separately archived, so byte-for-byte replay of that exact file version is unavailable.
- The executed training script snapshot is preserved in the readout directory and matches its recorded SHA. The tracked script is now SHA-256 `c30beaeac82cdfcd994fa1ff1ee8151e10d1f1f73babaf5e35e898795eb3245e1`; post-run edits only moved the local model import into the function and removed an ignored tokenizer `max_length` argument (tokenization had emitted a warning that the argument was ignored with `truncation=False`). The run metrics and checkpoint remain those from the hashed executed snapshot.

## Runtime, size, and artifacts

Feature extraction used the pinned JAX/TensorFlow worker with `JAX_PLATFORMS=cpu`
and hidden CUDA devices. It completed 336 images in 557.14 seconds (including
2.94 seconds model load), with JAX `TFRT_CPU_0`, Python 3.11.16, JAX 0.6.2,
TensorFlow 2.18.1, and NumPy 1.26.4. The worker reported 812,326,912 bytes RSS
after model load. The RTX 3080 was not used. Readout training used CPU PyTorch
2.6.0+cu124, Transformers 5.6.2, and NumPy 2.4.6; text feature extraction took
0.141 seconds and the eight-epoch readout loop 0.083 seconds. Peak RAM for the
readout process was not captured. Readout Python was 3.11.9, with
`CUDA_VISIBLE_DEVICES` empty. No paid/cloud compute was used.

Parameter count is within the provisional 100–200M target. Uncompressed weight
files are not within the provisional 200MB package target: VideoPrism checkpoint
457,475,296 bytes, MiniLM weights 69,569,488 bytes, best readout 592,888 bytes,
estimated combined weights **527,637,672 bytes**. No quantization or compression
was attempted.

External artifacts (not committed to Git):

- Frozen features and extraction metadata: `C:\CodexArtifacts\pretrained-reuse-r2\openimages-videoprism-features-20261026-attempt-01\`
- Readout, validation predictions, full eight-epoch history, and run metadata: `C:\CodexArtifacts\pretrained-reuse-r2\openimages-videoprism-readout-20261026-attempt-01\`
- Best readout SHA-256: `5e13b70b45eece770840f75db2fd5c70da3d3045897b2a35a9ddb0a623842e0c`
- Validation prediction SHA-256: `42cbfd4b5e0d68be87f2f2657e13e14c78166906d06262896f68149d255b1400`
- Executed training runner snapshot SHA-256: `776ebe99942d36281bf379933ff291b365a77761cba88f820f46ace348d81585` (`executed-runner.py` in the readout directory).
- Source commit recorded by the run: `b894e2b982ab3df8519552b8f4ce2d60c37ee56a`.

The extraction command used `scripts/extract_openimages_videoprism_features.py`
with the task manifests, both preserved media-cache attempts, pinned Path A
assembly/source, and the 3.11.16 VideoPrism virtualenv shown above. The readout
command used `scripts/train_openimages_frozen_probe.py`, the pinned MiniLM
folder and VideoPrism checkpoint, and an external new output directory; the
process environment set `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`, and
`MKL_NUM_THREADS=4`. Training metadata records start/end UTC timestamps and
all input/config/feature/model hashes.

The result supports continuing Path A with a task-specific image readout, not
promoting it. This probe does not yet package a standalone raw-image-to-JSON
inference runtime. Next, obtain a rights-cleared image evaluation with adequate
independent examples per class and a separate reporting split, then test
whether the same compact option interface transfers to independent tasks. Do
not claim the 95–98% objective from this pilot.
