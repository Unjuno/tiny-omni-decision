# TallyQA Visual Genome AMT Source Screen — 2026-11-10

## Disposition

`DEVELOPMENT_PROBE_COMPLETE_NOT_PRODUCT_ADMITTED`. A bounded natural-image
counting Decision task was constructed from pinned TallyQA training
annotations, using only AMT-origin rows and Visual Genome paths. This is a
text-conditioned image task; it does not satisfy the roadmap's text-free image
M-0 probe or any M-2 multimodal-fusion gate. Do not present it as general image
understanding. A follow-up found that the four-choice candidate set itself
leaks many count labels; retain the recorded model metrics as historical,
shortcut-confounded development results, not clean counting accuracy. See the
[integrity correction](R2_TALLYQA_OPTION_POLICY_INTEGRITY_20261112.md).

Do not read the official `test.json` labels. The official archive contains a
separate test member, but this screen opened only `train.json`; no official test
labels or test media were loaded.

## Pinned source and rights

- Dataset repository: [TallyQA_dataset at pinned commit
  `46cdc649ec79c3dcc2720ff227ad07d7ee51da6f`](https://github.com/manoja328/TallyQA_dataset/tree/46cdc649ec79c3dcc2720ff227ad07d7ee51da6f).
- The repository includes an Apache-2.0 license for its annotations/code. The
  exact pinned `LICENSE` SHA-256 is
  `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
- The official README says the archive mixes COCO and Visual Genome images and
  that `data_source` distinguishes AMT questions from imported VQA/TDIUC rows.
  This screen accepts only `data_source=amt` and `VG_100K` / `VG_100K_2` paths;
  all COCO paths and all imported/generated questions are excluded.
- Visual Genome's [publisher page states CC BY 4.0](https://homes.cs.washington.edu/~ranjay/visualgenome/about.html).
  This covers the selected VG source terms as stated by its publisher; it does
  not replace attribution or checks for duplicate/use overlap with prior work.
- The [TallyQA paper](https://cdn.aaai.org/ojs/4815/4815-13-7881-1-10-20190709.pdf)
  describes AMT questions as complex-counting questions and reports a separate
  Visual Genome-only complex test split. That test was not inspected here.

## Immutable input and observed data

| Item | Value |
|---|---|
| Official archive | `tallyqa.zip` from the pinned repository commit |
| Git blob SHA-1 / declared size | `3b4324bcecdbfec82c03e7b45ed926d8caca7c42` / 5,691,198 bytes |
| Downloaded archive SHA-256 | `e805199fa25c91d2f9df964920c6c36ce8d6cb941c970176f38a92f971bab5ea` |
| Pinned README SHA-256 | `a3452dba9af5d893df2528ce39489b5901f3565d040c73be43557e215a4c7bbf` |
| Training member size | 47,187,804 uncompressed bytes; 249,318 rows |
| Selected subset | 3,902 AMT rows, 3,494 unique `image_id` values, 3,494 distinct source paths |
| Selected label range | Integer answers 0–15 |
| Question text | 3,330 distinct NFKC/casefold/alphanumeric-normalized questions |
| Excluded rows | 245,416 other-source rows; no COCO images or imported/generated questions used |

The pinned README's example schema includes an `issimple` field. It is absent
from every record in the downloaded `train.json` member. This is an upstream
schema discrepancy; the screen therefore does not use that field or infer
simple/complex status from it. The AMT source filter is independently supported
by the paper's description, but per-example complexity is not re-labeled here.

## Frozen development split preflight

Split seed is 17. Each image is grouped with all its questions. In addition,
images sharing any normalized question text are unioned into the same component,
so no normalized question string crosses the split. A stable SHA-256 ordering
of components determines a greedy assignment to the 20% validation image
target. This is deterministic and does not use model outputs.

| Split | Unique images | Questions | Ordered question-ID SHA-256 | Sorted image-ID SHA-256 |
|---|---:|---:|---|---|
| Train | 2,795 | 3,101 | `6cfeaacbf7a04a9cfe8bb91ab4dc9677e64dbc9c311d528e4a89a49a8ce33fdb` | `cdf91a74eb40621246ded16f5f8f97037633ca14db1eab96441dcac7bc71904d` |
| Validation | 699 | 801 | `26d495577e2cc290b03522744cc20878b05ed76f10025e803fb5086b8cc4c9c5` | `b7579d1e3d4340191c6f4adf86eacd2d9f5adf87e93972035932368d1c6741db` |

There are 2,924 connected components; the largest contains 95 image IDs. The
computed split has zero image-ID overlap and zero normalized-question overlap.
The selected media were subsequently fetched, JPEG-decoded, SHA-256 hashed,
and verified to have zero duplicate bytes across the frozen train/validation
partition. This check is scoped to this task's 3,494 images; it is not a full
hash comparison against every retained local corpus/artifact or against
pretraining data of either frozen encoder.

The complete deterministic split manifest, including question/image IDs,
source-relative image paths and normalized-text hashes, is outside Git at
`D:\CodexArtifacts\pretrained-reuse-r2\tallyqa-vg-source-20261110-attempt-01\source-screen.json`
(the hash is recorded in the live run artifact; it changes if this manifest is
regenerated): SHA-256
`110fbbe4983106dceb90e82d5454aa40dd3eaebd6151fae03c8f6e6af40a516a`. The
source archive and pinned README/license snapshots are in the same artifact
directory. The screening utility verifies the pinned archive SHA and reads only
`train.json`:
[`scripts/screen_tallyqa_vg_train.py`](../../scripts/screen_tallyqa_vg_train.py).

All 3,494 selected images were acquired from the Visual Genome publisher and
verified as JPEGs: 2,795 train and 699 validation, 495,530,887 bytes total,
3,494 unique file hashes and zero train/validation byte-identity overlap.
The acquisition receipt SHA-256 is
`13707d624dcd3d9ee814b85436d5346f230845a095a5b8067bb15b396976f952`.
The two initial delivery samples and their hashes are recorded above in the
prior source-screen history. The publisher documents the image archives at
[its API page](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html).

## Frozen-feature development probe

**Subsequent integrity note (2026-11-12):** the four-option candidate set
uniquely reveals labels 3 through 13. The original 55.93%/56.30% measurements
below remain historical but are shortcut-confounded and are not clean counting
accuracy estimates. A separate answer-independent 16-choice follow-up scored
52.65% joint and 52.97% question-only on an inner development split drawn from
the old training pool; it is not blind evidence. See
[`option-policy correction`](R2_TALLYQA_OPTION_POLICY_INTEGRITY_20261112.md)
for the complete audit and follow-up.

The open-ended source labels were adapted into a project-created four-option
task: the correct integer count plus the three nearest distinct counts, with
fixed boundary distractors for answers 0, 1, and 15. Options were deterministically
permuted from seed 17 and question ID. This is a different task from the
official open-response TallyQA evaluation. The prepared manifest contains all
3,101 train and 801 validation questions, keeps every image/question group in
one split, and matches every image receipt. Its SHA-256 is
`59c1c03edf17a8da6c870d4827d033af58759c8300f953b60cb0abbff98f8a45`.

VideoPrism-B at revision `be719a406d563b66f0ac969e7c94bab8e997c81a` and
checkpoint SHA-256
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29` was
loaded from pinned source commit `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`.
The image was repeated over eight static temporal slots, center-cropped to
288px, mean-pooled to 768 float32 features, and extracted on the JAX CPU
backend. MiniLM-L3 at revision
`4ca70771034acceecb2e72475f72050fcdde4ddc` was also frozen. A 196,865-parameter
candidate MLP was trained jointly on image, question and supplied count;
the question-only control had 98,561 trainable parameters. Checkpoints were
selected by minimum NLL on the 801-question validation set. Attempt 01 saved
weights but failed its post-fit finite-metric serialization guard. Attempt 02
selected on training NLL due to a code-path defect. Both attempts and their
artifacts are preserved with audit markers and excluded from quality evidence.
Attempt 03 used independent validation inputs and passed reload-equivalence
checks.

The development run used source commit `e07e049ff4387484fb754e0fff963631f3cc7c85`
on `codex/r2-tallyqa-vg-source-screen-20261110`. The feature-cache process
started at 2026-11-10 19:09:14 JST and completed at 20:08:58 JST; measured
VideoPrism extraction time was 3,582.78 seconds. The final valid readout result
was written at 20:12:28 JST. The extract-to-result window was about 63 minutes;
image-download start time and separate MiniLM embedding duration were not
captured. The machine was an Intel Core i7-12700H / RTX 3080 Laptop GPU (16
GiB); VideoPrism ran on JAX CPU, and the readout recorded a 22.9 MB peak
allocated GPU memory after the text feature stage. No cloud compute or compute
cost was used.

Commands (from the repository root; external D: artifact path is intentional):

```powershell
$env:PYTHONPATH = "$(Get-Location)\src;$(Get-Location)"
$run = "D:\CodexArtifacts\pretrained-reuse-r2\tallyqa-vg-source-20261110-attempt-01"
$vp = "C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe"
& $vp scripts\prepare_tallyqa_vg_task.py --screen "$run\source-screen.json" --acquisition "$run\full-media\acquisition.json" --archive "$run\tallyqa.zip" --output "$run\task-manifest.json"
& $vp scripts\extract_tallyqa_videoprism_features.py --task-manifest "$run\task-manifest.json" --media-root "$run\full-media" --checkpoint C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-f16r288.npz --source-root C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source --output-dir "$run\feature-cache" --batch-size 4
& C:\CodexArtifacts\venvs\embeddinggemma2-v518\Scripts\python.exe scripts\train_tallyqa_vg_readout.py --config configs\pretrained_reuse\path_a_tallyqa_vg_videoprism_seed17.yaml --task-manifest "$run\task-manifest.json" --feature-cache "$run\feature-cache\features.npz" --cache-report "$run\feature-cache\cache-report.json" --text-model C:\CodexArtifacts\pretrained-reuse-r2\models\text-minilm-en --output-dir "$run\readout-attempt-03"
```

The config SHA-256 is
`60857ddebb2ab0d741d46f9e08b077160069d536c80dbd862654ef5b1c481164`.
Attempt 03 selected checkpoints using validation NLL and reloaded the saved
states for comparison; joint predictions matched before/after reload. The
readout implementation subsequently gained an extra pre-fit cache-to-task
identity guard. It does not change training computation; its CPU regression
test verifies mismatched image/hash pairs fail closed.

| Validation condition (801 questions) | Accuracy | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|
| Joint image + question | 55.93% | 1.0925 | 0.5915 | 0.0387 |
| Question-only trained control | 56.30% | 1.1226 | 0.6095 | 0.0577 |
| Joint checkpoint, zero image features | 50.56% | 1.2042 | 0.6526 | 0.1170 |
| Joint checkpoint, image features shuffled by image ID | 50.19% | 1.2166 | 0.6515 | 0.0725 |

Validation target option positions were 206 / 206 / 187 / 202 for positions
0 / 1 / 2 / 3, respectively.

On the validation clusters, the image-zero and shuffled-image Accuracy
difference from the joint condition was −5.37 and −5.87 percentage points;
the 2,000-replicate image-group paired-bootstrap 95% percentile intervals
were [−8.23, −2.66] and [−8.68, −2.96] points. Question-only minus joint
Accuracy was +0.63 points (interval [−2.17, +2.90], including zero); its NLL
was 0.0301 worse (interval [+0.0021, +0.0572]). The image-ablated comparisons
support that this readout uses visual features, but do not show a robust
Accuracy gain over a separately trained question-only model.

The learning curve overfit: joint training Accuracy/NLL changed from
62.56%/0.9094 at selected epoch 4 to 76.46%/0.5846 at epoch 16, while
validation Accuracy/NLL changed from 55.93%/1.0925 to 52.81%/1.3207. The
question-only model selected epoch 3 and ended at 64.88%/0.8633 train and
51.44%/1.1466 validation. Validation was repeatedly observed for selection;
bootstrap intervals are descriptive and not a blind confirmatory result.

| Epoch | Joint train Acc/NLL | Joint validation Acc/NLL | Question-only train Acc/NLL | Question-only validation Acc/NLL |
|---:|---:|---:|---:|---:|
| 1 | 49.34% / 1.1345 | 48.31% / 1.1700 | 51.76% / 1.1353 | 48.31% / 1.1753 |
| 2 | 57.14% / 1.0361 | 51.69% / 1.1437 | 57.85% / 1.0462 | 53.81% / 1.1354 |
| 3 | 61.50% / 0.9494 | 54.93% / 1.0953 | 59.34% / 1.0109 | 56.30% / 1.1226 |
| 4 | 62.56% / 0.9094 | 55.93% / 1.0925 | 59.85% / 0.9918 | 53.43% / 1.1234 |
| 5 | 61.75% / 0.9104 | 54.68% / 1.1439 | 60.63% / 0.9863 | 54.18% / 1.1320 |
| 6 | 63.79% / 0.8532 | 55.18% / 1.1180 | 61.27% / 0.9683 | 53.81% / 1.1231 |
| 7 | 64.72% / 0.8279 | 55.43% / 1.1257 | 61.04% / 0.9599 | 54.31% / 1.1338 |
| 8 | 64.88% / 0.8203 | 55.31% / 1.1714 | 61.79% / 0.9470 | 54.06% / 1.1266 |
| 9 | 67.27% / 0.7762 | 54.56% / 1.1446 | 62.33% / 0.9401 | 54.93% / 1.1439 |
| 10 | 68.78% / 0.7549 | 53.31% / 1.1595 | 61.50% / 0.9423 | 54.31% / 1.1562 |
| 11 | 70.36% / 0.7171 | 52.93% / 1.2062 | 62.30% / 0.9172 | 52.18% / 1.1377 |
| 12 | 70.46% / 0.7033 | 55.43% / 1.2308 | 63.11% / 0.9082 | 53.68% / 1.1502 |
| 13 | 74.49% / 0.6622 | 51.69% / 1.2116 | 63.30% / 0.8984 | 51.81% / 1.1445 |
| 14 | 74.81% / 0.6644 | 45.44% / 1.2580 | 63.88% / 0.8903 | 52.43% / 1.1456 |
| 15 | 75.81% / 0.6013 | 52.43% / 1.2701 | 64.17% / 0.8748 | 53.18% / 1.1570 |
| 16 | 76.46% / 0.5846 | 52.81% / 1.3207 | 64.88% / 0.8633 | 51.44% / 1.1466 |

The frozen resident parameter total is 131,951,873 (VideoPrism 114,365,184;
MiniLM 17,389,824; joint readout 196,865), before any quantization. This
parameter count is below 200M; the current unquantized encoder weights alone
exceed the roadmap's provisional 200 MB package target, so package-size
compliance is not established. Only readouts were trained. No ternary
quantization was performed. The sealed audit and official test member were not
loaded.

Reproducible run artifacts are outside Git at
`D:\CodexArtifacts\pretrained-reuse-r2\tallyqa-vg-source-20261110-attempt-01`:

- feature cache `feature-cache/features.npz`, SHA-256
  `ddd738a362d7b31762c676dc178fed0183ddce8052ffaac4bbc93f57ef2753a6`;
- extraction report `feature-cache/cache-report.json`, SHA-256
  `aa018cfcac4e714d26d43d7c7810193555dee7ebc02d848a6fa619bfe1e1df6d`;
- selected joint readout `readout-attempt-03/best-joint-readout.pt`, SHA-256
  `6a1ce691626d47a45ddf0a92e8c43c38e249de826c5f81b64a989e44ad85bb1b`;
- selected question-only readout `readout-attempt-03/best-question-only-readout.pt`, SHA-256
  `6c5773aaa10c133907a9592e6298ab7e8a31e4f44abd0904f7b7ba61894a97f9`;
- complete per-example validation outputs and epoch histories are in
  `readout-attempt-03/result.json`, SHA-256
  `9e1d88a83e35ef1455221234ac90296e453ae298af34a92f0501391c77c61e1f`.
- machine-readable summary in Git is
  [`r2-tallyqa-vg-development-20261110.json`](../../manifests/pretrained_reuse/r2-tallyqa-vg-development-20261110.json).

The 3,494 images were compared for duplicate hashes within this run, not
exhaustively against all older local project corpora or pretrained encoder
provenance. The repository has a TallyQA license-source entry, but that entry
does not prove prior payload use. This limits claims of globally novel media.
No official test or sealed audit labels were opened, no product model was
promoted, and this task-specific 56% result does not approach the project's
95–98% aspiration. It is a natural-image source/readout feasibility result,
not evidence that the overall objective is impossible.

## Next R2 gates

1. Continue with independent, fresh task-grouped text, audio and video
   development candidates; do not treat this text-conditioned image task as
   text-free visual quality evidence.
2. When combining modalities, enforce matched question groups, missing and
   shuffled controls, and calibration/utility metrics before any larger
   architecture.
3. Audit source-level and prior-artifact identity overlap more completely if
   this TallyQA adaptation is selected for further use.
4. Keep the sealed audit and official test untouched until a model-selection
   protocol and task battery are frozen.
