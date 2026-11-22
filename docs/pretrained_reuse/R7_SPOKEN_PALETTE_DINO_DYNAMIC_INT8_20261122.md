# R7 DINOv2 dynamic INT8 compatibility check — 2026-11-22

## Result and scope

This fixed CPU-only experiment applies PyTorch eager dynamic `qint8` to every
`nn.Linear` in the pinned DINOv2-small used by the controlled spoken-color
screen-action pilot. It tests one actual quantized encoder, strict state
save/reload, and the unchanged scorer on the pilot's already exposed
development evaluation set. There was no fitting, seed/config search,
temperature change, or checkpoint selection. This is not blind evaluation and
does not qualify a product model.

The DINOv2 state payload is 24,671,260 bytes (23.52 MiB), compared with
88,249,960 bytes (84.16 MiB) for the original float32 DINOv2 safetensors file.
That is a 72.0% reduction for this encoder payload. A fresh DINOv2 instance was
dynamic-quantized, loaded from the saved state with `strict=True`, and produced
finite 4-option features for all 24 evaluation screens. The fixed evaluation
contains 9,528 decision rows from four source-TRAIN speaker groups.

| Fixed development result | Float32 DINOv2 | Dynamic INT8 DINOv2 | Change |
|---|---:|---:|---:|
| Accuracy | 99.9895% | 98.6986% | -1.2909 pp |
| NLL | 0.00119 | 0.10804 | +0.10686 |
| Brier | 0.000409 | 0.036740 | +0.036331 |
| ECE-15 | 0.000863 | 0.071046 | +0.070182 |
| Mean confidence | 0.99904 | 0.91594 | -0.08310 |

The scorer's argmax changed on 123 of 9,528 examples. Maximum/mean absolute
probability changes were 0.9566/0.0432. The result shows that this encoder's
dynamic INT8 export can preserve high accuracy on this synthetic task while
materially changing probabilities and calibration. It does not meet a
probability-parity interpretation of quantization and must not be promoted
without a separately authorized, clean calibration and locked evaluation.

## Reproduction and artifacts

Run from the repository root with the existing pinned Python 3.11 environment:

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
$env:PYTHONPATH = "src"
python -m scripts.pretrained_reuse.evaluate_spoken_palette_dino_int8
```

The first implementation attempt stopped on the pilot/MiniWoB row-schema
boundary before encoding any feature; its partial output is retained at
`C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-dino-dynamic-int8-attempt-01`.
The corrected successful attempt is retained separately at
`C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-dino-dynamic-int8-attempt-02`.
Its `result.json` records all source, config, checkpoint, corpus, feature and
output hashes. The state payload SHA-256 is
`cd8083402b68ed41f9a3367258698ba7cc70c455b58f31828e29346e281381d8`.
Quantization took 0.316 s; inference over 24 unique screens took 1.518 s; the
full two-load experiment took 6.411 s on CPU with four threads. PyTorch was
2.6.0+cu124, using its `x86` quantized CPU engine. No GPU or cloud compute was
used.

## Limits and next engineering implication

- Audio input was a pinned, cached Whisper embedding bank; Whisper weights and
  audio preprocessing were not executed or quantized in this run.
- The 98,561-parameter Decision scorer and its scalar temperature remained
  float32 and unchanged. This is neither a complete multimodal export nor a
  package-size result; Python, PyTorch, tokenization/preprocessing, and runtime
  dependencies are excluded from the 24,671,260-byte encoder payload.
- The same exposed development examples were used only for fixed
  compatibility measurement. Their four independent speaker groups and
  generated screens do not establish natural audiovisual quality, generic
  image/video competence, or the 95–98% product targets.
- Quantization did not touch the preserved baseline checkpoint, corpus, or
  result. Sealed audit remained unopened.

This is an R7 feasibility datapoint only. It shows why Accuracy alone is not
enough to accept low-bit export: calibration and option probabilities changed
substantially. The next valid R7 gate still requires a complete, actually
packaged runtime with measured size/reload parity and device-specific latency;
R2's natural, rights-cleared both-input task gate also remains open.
