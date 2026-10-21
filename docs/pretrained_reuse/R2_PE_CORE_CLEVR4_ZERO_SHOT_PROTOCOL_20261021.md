# PE-Core T16 CLEVR-4 zero-shot transfer protocol — 2026-10-21

## Status and question

This is a one-pass, CPU-only development evaluation of whether the pinned,
pretrained PE-Core T16 image/text encoder can directly rank fixed CLEVR-4
taxonomy choices. It is not a blind test, a product result, or a replacement for
the learned Path A baseline. The 1,595-image CLEVR-4 reporting subset has been
used for prior reporting; it is not the sealed audit. No sealed audit is opened.

## Frozen inputs

- Model: `timm/PE-Core-T-16-384@7fe539ed578ac49a1c2b4f946e4b0747704c825a`.
- Converted weights: `open_clip_model.safetensors`, SHA-256
  `477b7507c79e3484195f4e389af6d044b5d56f42ce503d6b0eb7e99439b545e5`.
- Runtime: `open_clip_torch==3.3.0`, `timm==1.0.25`, CPU. Dependencies live in
  a separate `pip --target` directory; the project environment and dependency
  files are unchanged.
- Data: the existing CLEVR-4 official validation reporting slice from
  `image-videoprism-clevr4-full-seed17`; exactly 1,595 unique image IDs and
  6,380 decisions, four taxonomies per image. Read only row ID, image ID,
  taxonomy, target and options from the saved JSONL; ignore its old prediction
  and probability fields. Use the existing local PNG bytes read-only.
- Ordering: preserve first-seen image order from the saved reporting JSONL;
  within each image use `color, texture, shape, count`. Preserve each row's
  supplied ten-option order. The runner hashes ordered image IDs, decision IDs,
  and every image's bytes before inference.

## Fixed prompting and scoring

No prompt search, model fitting, checkpoint selection, temperature fitting or
calibration is allowed. Encode each option once with these exact templates:

| Taxonomy | Template |
|---|---|
| color | `In this synthetic 3D image, an object's color is {option}.` |
| texture | `In this synthetic 3D image, an object's surface texture is {option}.` |
| shape | `In this synthetic 3D image, an object's shape is {option}.` |
| count | `In this synthetic 3D image, there are {option} objects.` |

Apply the pinned OpenCLIP image preprocessing. L2-normalize image/text
embeddings and use `softmax(exp(checkpoint logit_scale) * cosine similarity)`
over the ten supplied options. No option shuffle or prompt ensembling.

## Required reporting

Report Accuracy, NLL, multiclass Brier, 15-bin ECE and mean confidence by
taxonomy and as an unweighted four-taxonomy macro. Save every prediction and
probability. Use a seed-17, 10,000-resample paired-by-image bootstrap (all four
taxonomy rows stay in each sampled image group) for macro Accuracy/NLL/Brier
95% percentile intervals; ECE remains descriptive. Record model parameters,
weight hash, exact runtime versions, input hashes and CPU time.

Only after the prediction artifact is written may historical Path A metrics on
the same rows be displayed as a contextual comparison. That comparison is not
used to alter prompts, model weights or calibration. Regardless of outcome,
this remains a development result on a previously observed source/split and
does not establish natural-image quality, multimodal complementarity, audio or
video capability, nor any 95% product gate.

## Runner and output

Runner: `scripts/evaluate_pe_core_clevr4_zero_shot.py`.

Output directory: `C:\CodexArtifacts\pretrained-reuse-r1\pe-core-t16-r1-attempt-03-clevr4\`.

No training, GPU use, media mutation, audit access, or change to existing model
artifacts is authorized by this protocol.
