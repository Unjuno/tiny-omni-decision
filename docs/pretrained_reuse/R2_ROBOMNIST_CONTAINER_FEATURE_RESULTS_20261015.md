# Native container invariance and frozen pretrained features — executed

This completes an input/frontend mechanics gate on one known training-side
RoboMNIST recording. It does **not** establish native decision accuracy, temporal
reasoning, multimodal complementarity, a blind result, or the full Omni goal.

## What ran

The prospectively committed [protocol](R2_ROBOMNIST_CONTAINER_FEATURE_PROTOCOL_20261015.md)
uses fixed pilot `Ar1_Vhigh_Ac0_Rp01_UNC062`, all three camera files, and one
1,048,576-byte zero `free` payload plus8-byte header per new derivative. Original
source prefixes/files and all previously executed raw/length/model evidence
are protected. No reencoding or repeated padding/model selection.

Producer source `3e91eef37a1d53ed8ab768e68c2f396cb21f7fe4`:

```powershell
python -m scripts.probe_robomnist_container_features --output C:/CodexArtifacts/pretrained-reuse-r2/robomnist-container-feature-smoke-01
```

Fresh consumer source `127c1aa48f04146cae57984caef97e017d1413bf`:

```powershell
python -m scripts.replay_robomnist_container_features --main C:/CodexArtifacts/pretrained-reuse-r2/robomnist-container-feature-smoke-01 --output C:/CodexArtifacts/pretrained-reuse-r2/robomnist-container-feature-replay-01.json
```

CPU Intel Core i7-12700H,2threads,float32,eager attention,seed17. Existing
PyTorch2.6.0+cu124/Transformers5.6.2/NumPy2.4.6/PyAV18.1.0/Pillow10.4.0,
no environment upgrade. CUDA remained uninitialized; no GPU or cloud compute.
All resident83,101,185 parameters, including unused text tower, are frozen.
Parameter bytes332,404,740; the full checkpoint is332,443,148 bytes. This prototype
does not meet the200MB package target and is not a claimed38M deployment.

Model: `wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M`
revision `07a4b0bc751cb64fecd2b661c048c1dd98d69444`, weights
SHA `112bdfb07ebc8994e9afba7a2a314e278493a675033595cf48e54cf8a17ae587`.
Released weight rights are MIT; data attribution remains
[RoboMNIST v1, CC BY4.0](https://doi.org/10.6084/m9.figshare.28179383.v1).
The dataset's rights do not confer a license on upstream code.

## Actual results

| Check | Measured result |
|---|---|
| All native frames/PTS |450 per derivative,1,350 total, all RGB frame hashes/PTS/metadata equal original |
| Retained observations |3 cameras ×3 fixed times ×2 full lenses =18 |
| Original vs derivative tensors/features |Every byte identical |
| Uniform mean-color control |All18 feature vectors differ; L2 range6.5470–6.9436 |
| Complete model state |Before/after SHA `061d84f192ef4529c6f0a59f82f9287eeb450c7327fa47967998e0c673674bb5` |
| Training/gradient/RNG |0 trainable, no gradients, eval mode, forward CPU RNG unchanged |
| Producer |25.26494s; peak1,175,064,576 bytes;126 evidence pins |
| Separate fresh replay |37.95437s; peak1,229,021,184 bytes;134 pins; all8 arrays exact |

Producer starts `2026-10-14T17:08:52.617493+00:00`, ends
`2026-10-14T17:09:17.881982+00:00` (JST2026-10-15). The replay independently
reopens **both** originals and derivatives (2,700 full frames), loads a new model
and recomputes all original/derivative/uniform pixel tensors and512-D vectors.
Each process encodes54 images in18 forward calls. It shares the frontend
implementation with the producer; it is a fresh execution, not an independent
implementation of that encoder. Eight saved arrays include all six pixel/feature
arrays and two complete60-class length-control log-posterior arrays.

Frames contain horizontally concatenated stereo views. Each complete1280×720
lens is independently letterboxed to224² before native normalization, avoiding
the standard center crop at the stereo seam. Ordered features retain time/view
dimensions `[3,3,2,512]`; this is **per-frame image encoding**, not a learned
video encoder or evidence that it understands motion.

### Fixed nuisance classifier comparison

The original1,475-train-group centroids are reused without fitting, calibration,
hyperparameter changes or validation selection. This is one known train row.
IDs are original60-class indices; complete posterior vectors are saved.

| Length-only arm | Original top1 | Derivative top1 | Posterior L1 change |
|---|---:|---:|---:|
| camera_length |0|48|1.646940|
| all_lengths |9|22|1.550488|

Thus harmless container bytes change these two nuisance predictions while native
decoder pixels and frozen visual features remain identical **on this recording**.
It does not prove that learned features are free of visual source shortcuts, or
that all containers/decoders tolerate this intervention. Sensitivity to a uniform
image only proves a pixel-dependent representation here, not correct decisions.
No Accuracy/NLL/Brier/ECE is claimed from this mechanics probe.

## Integrity corrections and saved evidence

Preexecution review found that an editable R1 report could repin different
weights. Historical audit contents are now checked before parsing, with an
independently fixed release weight SHA. Tests reject repinned altered weights.
Consumer script/test hashes are captured before long decoding/model work and
rechecked at completion, so a later edit cannot replace the executed provenance.
Full source/model/library/product/report hashes are verified at completion,
followed by CPU/time/disk bounds. No historical producer or artifact was edited.

Main result SHA:
`12a20f436a487e4e21a91956d58bf1fb9bd231ef1494eb9abc8ca7346425ddba`.
Config SHA `406abd9ea230d3bf5fb24f10f30d5992ef1a5d6d8dac1171b5c29b1e03a19ea1`.
Arrays SHA `7800a959fb08f2d0f8a8c5b7c26d26d5f5454634a76c6d56fe1ffdf40960116e`.
Fresh replay SHA `40b5d246ad6fb5e28220ff5565be7328be97cb24a6a5115d86d0eaae5f1657fc`;
consumer SHA `132ebea7ce416f5de094106b61efab90764518a866ea8d3db4533562b698146b`.
The [small JSON](R2_ROBOMNIST_CONTAINER_FEATURE_RESULTS_20261015.json) retains
measurements, artifact pins, camera sizes and all18 uniform differences;
[CSV](R2_ROBOMNIST_CONTAINER_LENGTH_CONTROL_20261015.csv) retains both controls.
Large weights/media/tensors remain outside Git in the command paths above.

Verification: Ruff passes; CPU tests1,497 passed/1 skipped; simulated absent-Torch
tests1,083 passed/96 skipped; all existing model/dataset/catalog manifests and
dataset audit pass. Actual model tests run separately from normal CPU CI.

## Remaining gates and next work

One recording cannot establish class coverage, session generalization or joint
benefit. Common audio/video/CSI capture clock remains **UNVERIFIED**; the corpus
remains **NOT_ADMITTED** for joint training. Metadata-derived development results
remain exposed and are not recast as sealed evaluation. Position GT, legacy audit
and legacy Teacher/GPU artifacts were not opened or changed.

Next fix a useful public-data task contract and admit genuinely native grouped
inputs before fitting; retain length/no-input/evidence-destruction controls.
Synchronization, audio/sensor representations, complementary joint decisions,
native temporal/conditional semantics, full resident package cost and all four
modality/OOD95–98% requirements remain open. No compression or product promotion.
