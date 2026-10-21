# PE-Core T16 zero-shot CLEVR-4 result (2026-10-21)

## Scope

This is a one-pass development evaluation of the pinned PE-Core T16 image/text
encoder on the previously observed CLEVR-4 reporting subset. It is not a blind
audit and does not support prompt selection, calibration, or product promotion.
The predeclared protocol is in
[`R2_PE_CORE_CLEVR4_ZERO_SHOT_PROTOCOL_20261021.md`](R2_PE_CORE_CLEVR4_ZERO_SHOT_PROTOCOL_20261021.md).

No model weights were trained or modified. The model used was
`timm/PE-Core-T-16-384` revision
`7fe539ed578ac49a1c2b4f946e4b0747704c825a`, converted safetensors SHA-256
`477b7507c79e3484195f4e389af6d044b5d56f42ce503d6b0eb7e99439b545e5`, with
69,546,113 runtime parameters and OpenCLIP 3.3.0. The run used CPU, two Torch
threads, and did not initialize CUDA. Model and inference time was 125.815 s.

## Data and artifacts

- 1,595 images / 6,380 decisions; image-group bootstrap, seed 17, 10,000 draws.
- Image-ID-order SHA-256: `0b5170ebb1256bbd7bb79d36e487738149eee28ef32ed137ee8346c068156dc4`.
- Decision-ID-order SHA-256: `83cf145457c0966af7486be93b7c67f4b1576190638a5ac07dadae648cac72bc`.
- Result JSON: `C:\CodexArtifacts\pretrained-reuse-r1\pe-core-t16-r1-attempt-03-clevr4\result.json`.
- Result SHA-256: `ba59f48e08cc2829897a580b4820e5dee46686a5344f9036c3648e425c972e29`.
- Full per-example predictions and probabilities are retained in that JSON.

## Metrics

| Task | Accuracy | NLL | Brier | ECE-15 | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Color | 61.13% | 1.2874 | 0.5682 | 0.1079 | 0.5051 |
| Texture | 17.81% | 2.4212 | 0.9596 | 0.2580 | 0.4359 |
| Shape | 74.29% | 0.9761 | 0.4052 | 0.2008 | 0.5476 |
| Count | 17.99% | 2.2272 | 0.8853 | 0.0580 | 0.1220 |
| Macro | 42.81% | 1.7280 | 0.7046 | 0.1562 | 0.4026 |

Image-group bootstrap 95% intervals for macro Accuracy, NLL, and Brier were
`[41.72%, 43.86%]`, `[1.7069, 1.7495]`, and `[0.6965, 0.7127]` respectively.

## Context against existing Path A

On the same reporting IDs and supplied options, the historical Path A predictions
had macro Accuracy 88.06%, NLL 0.2912, Brier 0.1536, ECE 0.0177, and mean
confidence 0.8693. Its task accuracies were color 99.44%, texture 91.85%, shape
99.75%, and count 61.19%. Candidate-minus-Path-A paired image-cluster bootstrap
deltas were Accuracy −45.25 points (95% interval −46.49 to −44.03), NLL +1.4368
(+1.4134 to +1.4596), and Brier +0.5510 (+0.5411 to +0.5609).

This contextual comparison does not make the two model paths causally matched;
the existing Path A predictions are historical, and the cohort was already
observed during development. The result shows that direct zero-shot PE-Core
similarity with these fixed prompts/options is inadequate for this CLEVR-4 task.
It does not establish that the representation cannot support a trained small
readout, nor that PE-Core is unsuitable for other modalities or tasks. The weak
texture and count results identify concrete failure modes; no prompt tuning was
performed in response.

## Next bounded question

If a separate, predeclared frozen-feature readout experiment is run, it must use
a new output directory, preserve this result and the existing Path A artifacts,
select only on the existing 512-image checkpoint-selection split, and evaluate
the reporting subset once after its configuration is frozen. Use the cooperative
HotUGPU reservation wrapper for GPU work. At the 2026-10-21 follow-up snapshot,
the RTX 3080 Laptop showed 15,821 / 16,384 MiB used and 33% utilization; no GPU
work was started from that occupied state. No quality claim or product adoption
follows from this zero-shot result.
