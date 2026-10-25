# CLEVR compositional option-task preflight

**Status: source and input-path preflight only. No fitting, checkpoint
selection, quality evaluation, or sealed-audit evaluation has run.**

## Why this is a distinct task

The completed CLEVR-4 probes use fixed taxonomies such as color, texture,
count, and shape. This candidate instead takes a CLEVR image and its natural
language question together, then returns probabilities over the options
provided with that question. The question families cover attribute queries,
counting, comparison, spatial relations, and logical operations. This tests
conditional image-and-text decisions and compositional reasoning, not another
readout on the already exposed CLEVR-4 labels.

The official CLEVR page describes 70,000 train images / 699,989 questions,
15,000 validation images / 149,991 questions, and a test set whose answers are
not released. It lists the dataset under CC BY 4.0. The official train and
validation image identities are separate. See the [CLEVR dataset page](https://cs.stanford.edu/people/jcjohns/clevr/).

## Source and local artifact audit

| Source | Pinned evidence | Access in this preflight |
|---|---|---|
| CLEVR v1.0 annotation archive | S3 version `72l1MZGsQclDTzwW7vRQeXBB9sY3Jnsu`; ETag `"3fdac7d8fb7aff710dd69a98a06dd739-6"`; 89,363,837 bytes; SHA-256 `D4435DCEAB30022B64F7D34B3EC217162BFC1C2950C79999784750CED22674C4` | Downloaded to `C:/CodexArtifacts/pretrained-reuse-r2/clevr-vqa-source-preflight-20261025/CLEVR_v1.0_no_images.zip`; archive lists annotations and license, with no image payloads. |
| CLEVR v1.0 image archive | S3 version `B2HsawCh6GoabZXQDbLoboNXB4FrlnvC`; ETag `"3426f0a360618fcf2179966a8d1428c3-1210"`; 19,021,600,724 bytes; HTTP `Accept-Ranges: bytes`; 100,015 central-directory entries | Metadata plus two selected train/validation PNG payloads downloaded by byte range. Full archive not downloaded. |
| CLEVR-CoGenT v1.0 annotations | S3 version `1pgXk1N966vKCADkX4VK8BtdpNFyzYXD`; ETag `"1900bbeda991856079316bcc7c464138-8"`; 110,502,280 bytes; SHA-256 `1B3A7DE7624E4204CEA9EB91C9125EBCEBEA8E92A9088606F6114AB39CEDB9E5` | Downloaded to the same preflight directory. The ZIP central directory was listed; its question/answer members were not opened. |
| CLEVR-CoGenT v1.0 image archive | S3 version `haDYjDZdal8NjYimkkMSeCanaOmhAc5y`; ETag `"a2125a3153354b3b09849befcdd79ad1-1572"`; 24,719,655,191 bytes; HTTP `Accept-Ranges: bytes`; 130,020 central-directory entries | Metadata-only range inspection found 15,000 `valB` images (2,834,776,362 compressed bytes). No media or answer payload was read. |

The source archives identify CLEVR as CC BY 4.0. The full image archives are
large, but the inspected ZIP metadata permits selective byte-range retrieval.
The official train+development media set totals 16,073,098,963 compressed
bytes; the CoGenT-B validation images total about 2.83 GB. Current C: free
space was 38.63 GiB at the initial metadata screen. No bulk media has been
cached. A later CPU media-path smoke fetched one selected train image and one
selected validation image through conditional byte ranges, checked local ZIP
headers/CRC and wrote them to a separate two-image directory. It did not fetch
CoGenT-B.

The annotation pass briefly explored a seed-17 split inside CLEVR train and
decoded answer fields from its proposed 10% reserve while counting labels. I
stopped it before any metrics, manifests, or report were emitted. That reserve
is **not** a blind audit and will not be used as one. The official CLEVR
validation split is development data; the parser decoded its records to check
image grouping but did not inspect or aggregate its answers. CoGenT-B
validation remains unopened and is the prospective project-held-out one-time
audit after the model and selector are frozen. No test answers are available
or used.

## Candidate contract

- Input: one CLEVR image, its question text, and supplied candidate answers.
- Target: the source answer's index in the supplied option list.
- Fixed answer kinds: boolean (`yes/no`), size (`small/large`), material
  (`metal/rubber`), shape (`cube/cylinder/sphere`), color (eight CLEVR colors),
  and count (`0`–`10`).
- Deterministic choice construction: include the gold answer, add up to three
  distinct distractors from the same answer kind using seed 17 keyed by
  image/question identity, then deterministically shuffle option order. This
  yields two to four options depending on the answer-kind vocabulary. No
  question, scene, or media identity may cross the official train/validation
  split.
- Metrics: Accuracy, NLL, Brier, and ECE over supplied options; macro results
  across the six answer kinds; and per-kind and per-outer-program results.
  The implementation also reports a uniform-choice baseline and paired
  image-question controls (same question with a fixed gray image and with a
  different image from the same split). The normal-image validation NLL alone
  selects the checkpoint; controls are diagnostics only. No calibration or
  option-wording search on CoGenT-B.
- Selection: fixed-validation macro NLL, with a predeclared tie-break on macro
  Brier. Preserve the learning curve and best checkpoint separately from the
  terminal checkpoint.
- Audit: one evaluation on CoGenT-B validation after preprocessing, training
  schedule, and selector are frozen. Run the full split and all predeclared
  controls together; never feed its scores back into another model choice.

## Candidate checkpoint and measured forward

The local ViLT VQAv2 checkpoint is `dandelin/vilt-b32-finetuned-vqa` at
revision `d0a1f6ab88522427a7ae76ceb6e1e1e7b68a1d08`, with checkpoint SHA-256
`4d5f3409947b0369487ece7c5868f0040ceb67d25735dbb4ac5e99e03bab3a19` and
117,588,537 parameters (111,595,008 ViLT parameters plus a 5,993,529-parameter
3,129-class VQA head). The model hub declares Apache-2.0; the model card does
not document enough VQAv2 training-data provenance to close that separate
rights review. At the exact pinned model revision, the card says it was
fine-tuned on VQAv2, disclaims authorship by the ViLT team, and leaves training
data, preprocessing, and evaluation sections as "to do". The upstream
repository's DATA.md says it does not redistribute datasets because of
licensing issues. These statements do not identify the specific permissions
for every pretrained corpus. Links: [pinned model snapshot](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa/tree/d0a1f6ab88522427a7ae76ceb6e1e1e7b68a1d08), [pinned model card](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa/raw/d0a1f6ab88522427a7ae76ceb6e1e1e7b68a1d08/README.md), and [upstream DATA.md at commit 3db8b5035464afee84d951bf6322e1b27f1d072d](https://github.com/dandelin/ViLT/blob/3db8b5035464afee84d951bf6322e1b27f1d072d/DATA.md). Keep this a research-only probe; this metadata is not a product-rights clearance.

Its published classifier cannot directly represent all CLEVR answers: the
local label map includes all counts, yes/no, size, and material answers, but
only `cylinder` of the three shapes and seven of eight colors (no `cyan`). Do
not score a filtered subset and call it full CLEVR. The candidate Path-A test
is to freeze the ViLT representation and replace the closed head with a
28-answer linear readout (estimated 111,616,540 total parameters), then
softmax only the logits for the options supplied to each decision. The
candidate option-mapping and restricted-softmax utility is implemented in
`src/tiny_omni_decision/clevr_vqa_options.py`. A frozen-feature 28-answer linear
readout, same-kind option loss, per-kind/program metrics, best-versus-terminal
checkpoint handling, and a one-question-per-image corpus selector are in
`clevr_vqa_readout.py`, `clevr_vqa_probe.py`, and `clevr_vqa_dataset.py`.
The train/dev subset was generated on 2026-10-25 from the pinned official
annotation archive: 16,384 train images/questions and 4,096 official validation
images/questions, with zero train/validation image overlap. It remains a
development corpus; all six answer kinds occur in both splits. Validation
selection excludes any normalized question text selected for train, and the
final manifest records zero image and normalized-question overlap. Full media
extraction and feature caching are still pending. The source manifests
and metadata are outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2/clevr-vqa-subset-seed17-attempt-03/`.
Earlier attempts remain preserved and are superseded by attempt-03.

The two-image media smoke is at
`C:/CodexArtifacts/pretrained-reuse-r2/clevr-vqa-media-smoke-20261025-attempt-01/`.
Both image bytes came from the pinned ETag via HTTP 206 ranges; ZIP member CRCs
matched, and the recorded image SHA-256 values are
`f3fdf7e132f7f1b26df1fab872eb27b49eb3506f1f1d5e813cd0f68fec4a0b88` (train)
and `af87296b5d7c6c41d502bcb4c7d7a27b01efefd5df4ebb2292407913c7000c5a`
(validation). The frozen-feature CLI then loaded all 117,588,537 ViLT
parameters on CPU and emitted finite float32 features with shape `[2, 768]`
in 0.455 seconds of feature extraction. Feature SHA-256:
`8f90941d0501bddfec7c888dc24098319cf3a8dce5b35cd24f7d04078de9f36a`.
This is an input-path smoke only: it did not calculate answer metrics or fit a
readout. Its feature and metadata files are under
`C:/CodexArtifacts/pretrained-reuse-r2/clevr-vqa-feature-smoke-20261025-attempt-01/`.

A separate four-image CPU controls smoke used two train and two validation
examples. It produced finite `[4,768]` features for the original, fixed-gray,
and same-split shuffled-image inputs; the Sattolo permutation had no fixed
points or split crossings. Feature extraction took 1.793 s. The original,
blank, shuffled, and permutation-map SHA-256 values are
`555af155963d37f8e0ce9d99d91e3a5cd7c49ecc5c608724e5d87353c7e31c38`,
`684e85e61cf1fb6fba8446df5bf51910bc5b40e3f611d416d2a483ef62b80f75`,
`0de339489fee4303ed8709a846eedb302021eca68d6895b7c5e4c277b8b008ce`, and
`ea537c1f81bb8e7b5262c17accd84c572fcd02715aaff17c7d691d6671c666a7`.
It computed no quality metrics. Its files are under
`C:/CodexArtifacts/pretrained-reuse-r2/clevr-vqa-controls-feature-smoke-20261025-attempt-01/`.

The source archive SHA-256 is
`d4435dceab30022b64f7d34b3ec217162bfc1c2950c79999784750ced22674c4`.
`train-examples.jsonl` SHA-256 is
`70985cf8a95ade7fd7c5a5b969df9e7b4de1b30087d22c1f2b6949517937205c`;
`validation-examples.jsonl` SHA-256 is
`ed3ea2d6af987a3486860f4a4766b18fd309905626e49479561675883eef0d82`.
Train/validation answer-kind counts are respectively boolean 6,704/1,708;
color 1,397/366; count 3,948/953; material 1,401/339; shape 1,490/384; and
size 1,444/346. Outer question-program counts and option-cardinality counts
are saved in attempt-03 metadata.
The small readout has 21,532 trainable parameters; replacing ViLT's original
3,129-answer classifier yields 111,616,540 total parameters including the
frozen encoder. The existing FP32 checkpoint is 470,435,927 bytes, above the
provisional 200 MB package target; no quantization has been measured.

A separate CPU-only smoke used one official CLEVR train image and its paired
question. The range reader verified HTTP 206 and exact `Content-Range`; ViLT
loaded from the pinned local checkpoint and returned finite `[1, 3129]` logits
for a `[1, 15]` token input and a 480×320 image. Result: `PASS_CPU_FORWARD_ONLY`.
It did not read or use the answer, calculate quality metrics, use a GPU, or
touch CoGenT-B. The smoke script is
`C:/CodexArtifacts/pretrained-reuse-r2/clevr-vqa-source-preflight-20261025/cpu_forward_smoke.py`.

The prior ViLT result on CLEVR-4 (81.10% over its fixed 10-option image
taxonomies) is not a baseline for this question-conditioned task and will not
be mixed into its metrics. See [the earlier fixed-taxonomy report](R2_VILT_IMAGE_RESULTS_20261012.md).

## Current execution gate

The RTX 3080 Laptop was observed at 16,124/16,384 MiB and 99% utilization,
with PIDs `15928` and `26920` reported by `nvidia-smi`. At 08:50 JST, read-only
process inspection identified them as the HotUGPU supervisor and child for
`teacher-quality-clean-dev-v2-resume-20261025-attempt-01`; the child command
resumes the existing clean-dev-v2 checkpoint at step 896. Its latest persisted
training-history record was step 1,408, while trainer state/best remained at
step 1,280 (trainer-state timestamp 07:59 JST, history timestamp 08:18 JST).
The step-1,408 record showed 8.70 s/step, LR 1.18e-5, and gradient norm 42.28;
this is a rolling one-step training metric, not fixed-set generalization.
No new GPU job was started, and the live run and its code/artifacts were not
changed. The full CPU
suite passes (2,418 passed, 11 skipped), and Ruff passes. The ViLT input and
controls smokes ran on CPU. A matched full feature/readout run must wait for
GPU capacity.
CPU unit tests use synthetic pooled features and do not constitute
model-quality evidence.
There is no quality result yet, no product promotion, and no claim toward the
95–98% objective.

The first push's three GitHub Actions runs failed during `Test`. The CPU-only
CI environment installs the `dev` extra without optional PyTorch; the new
annotation-only dataset module had imported it transitively. PyTorch is now
loaded only in the probability routine that requires it, with a subprocess
regression test that blocks all `torch` imports while loading the dataset
builder. The corrected push passed all three public CI runs: [PR branch run
40143569784](https://github.com/Unjuno/tiny-omni-decision/actions/runs/40143569784),
[continuation run
40143566335](https://github.com/Unjuno/tiny-omni-decision/actions/runs/40143566335),
and [second PR event run
40143566252](https://github.com/Unjuno/tiny-omni-decision/actions/runs/40143566252).
A fresh isolated environment with only the `dev` extra and no PyTorch also
passed Ruff and pytest (1,808 passed, 154 skipped).

The manifest builder, selective media downloader, frozen-feature extractor,
and CPU readout runner are implemented. The approved 20,480-image acquisition
is running from the pinned ZIP using 8 workers and a 2 GiB disk reserve. At
2026-10-25 08:50 JST, 2,591/20,480 files (all validation so far) were saved and
verified. They are outside Git at
`C:/CodexArtifacts/pretrained-reuse-r2/clevr-vqa-media-seed17-attempt-01/`.
At 08:53 JST, a separate CPU verification found 3,177 downloaded validation
PNGs, all named in the pinned validation manifest, all decodable at 480x320;
none failed image verification. The manifest SHA-256 remained
`ed3ea2d6af987a3486860f4a4766b18fd309905626e49479561675883eef0d82`. This is
a partial-cache integrity check, not feature or quality evaluation.
At 09:00 JST, all 4,096 validation images had arrived; an independent CPU pass
matched every filename to that manifest, decoded every PNG at 480x320, and
found zero missing, unexpected, or invalid files. The training portion of the
download had then begun (66/16,384 images); the full cache remains incomplete.
At 09:07 JST, the train cache had 1,107/16,384 images. The Teacher v2 resume
was live with rolling history through step 1,495 and persisted best step 1,408.
An older, separate full-extent fresh-coverage GPU queue (supervisor PID 6080,
worker PID 33712; started 06:52 JST) was also still alive, waiting for the
device to become fully idle before its smoke/train/replay stages; none of its
three output directories existed yet. Preserve this earlier queue and let it
take the next GPU slot before submitting the CLEVR feature extraction.
At 09:12 JST, a CPU rescan covered 1,834 downloaded train PNGs: all names were
present in the frozen train manifest, all files decoded at 480x320, and none
failed verification. The train manifest SHA-256 remained
`70985cf8a95ade7fd7c5a5b969df9e7b4de1b30087d22c1f2b6949517937205c`; 14,550
selected train images were still pending.
At 09:14 JST, a second rescan covered the now 2,136 downloaded train PNGs
with zero unexpected names or decode/dimension failures; the manifest hash was
unchanged. These are prefix checks while acquisition continues, not a complete
train-corpus gate.
At 09:16 JST, a third rescan covered 2,524 train PNGs with the same manifest
hash and zero filename/decode/dimension failures; 13,860 selected train images
were not yet downloaded. The live Teacher resume and prior full-extent GPU
queue were still present. Its persisted training history had last advanced to
step 1,536 at 09:13:49 JST; no claim is made about the operation since then.
At 09:19 JST, a fourth CPU rescan covered 3,033 train PNGs with zero
unexpected names or decode/dimension failures and the same manifest hash;
13,351 selected train images remained to be fetched.
At 09:21 JST, the rescan covered 3,372 downloaded train images; all filenames
were in the frozen manifest and all PNGs decoded at 480x320, with zero
failures. This is still only a partial-media integrity check.
At 09:23 JST, the verified prefix grew to 3,660 train images, again with zero
unexpected filenames or image decode/dimension failures and the same manifest
hash; 12,724 train images remained to fetch.
At 09:27 JST, the latest scan covered 4,287 train images with zero unexpected
names or decode/dimension failures and the same manifest hash; 12,097 train
images remained.
At 09:31 JST, a CPU verification covered 4,950 train images; every filename
belonged to the frozen manifest and every PNG decoded at 480x320. No invalid
files were found; 11,434 train images remained to download.
At 10:18 JST, a CPU verification covered 12,126 train images, with zero
unexpected files or decode/dimension failures and the frozen manifest hash;
4,258 train images remained. The Teacher v2 resume process and older GPU queue
were still alive. The latest training-history row was step 1,664 (file updated
10:10:49 JST), while durable trainer state still recorded global/best step
1,536 and sample index 6,144. The process retained GPU memory and reported
varying device utilization, so no inference is made about its current
operation and it was left untouched.
The host exposes Intel Iris Xe Graphics and an RTX 3080 Laptop; Windows reports
1 GiB `AdapterRAM` for the Intel adapter. The active Python 3.11 environment is
PyTorch 2.6.0+cu124 with one CUDA device; `torch_directml` is absent and
ONNX Runtime exposes only Azure and CPU providers, not DirectML. There is no
verified secondary-GPU extraction path in this environment.
Next: finish media acquisition, extract full ViLT features and controls through
the cooperative HotUGPU reservation, fit the readout, then assess per-answer
and per-program metrics without opening CoGenT-B. Keep the prospective held-out
split sealed until model selection is frozen.
