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
  Report a uniform-choice baseline and paired image-question controls (question
  with a blank image and a shuffled image). No calibration or option-wording
  search on CoGenT-B.
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
rights review.

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
with PIDs `15928` and `26920` reported by `nvidia-smi`. No new GPU job was
started, and the live run and its code/artifacts were not changed. The full CPU
suite passes (2,416 passed, 11 skipped), and Ruff passes. The two-row ViLT
forward ran on CPU. A matched feature/readout run must wait for GPU capacity.
CPU unit tests use synthetic pooled features and do not constitute
model-quality evidence.
There is no quality result yet, no product promotion, and no claim toward the
95–98% objective.

The manifest builder, selective media downloader, frozen-feature extractor,
and CPU readout runner are now implemented. Next: acquire only selected
official train/development media, verify each media hash, then extract ViLT
features through the cooperative HotUGPU reservation and fit the readout. Add
paired blank-image/shuffled-image controls before selecting a candidate. Keep
CoGenT-B unopened through model selection.
