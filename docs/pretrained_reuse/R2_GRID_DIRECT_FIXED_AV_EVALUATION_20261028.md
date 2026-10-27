# R2 GRID corrected fixed AV evaluation (2026-10-28)

Status: **PASS for execution and integrity; weak, poorly calibrated performance
on one development-only speech/video correspondence probe.** This does not meet
the R0 quality aspiration and is not evidence of general AV reasoning.

## Scope and safeguards

The invalid original selection (`b33e4e7e...ccbf0d`) remains an unevaluated
integrity failure. This measurement used the replacement selection
`grid-av-direct-official-pilot-20261028-attempt-03/selected.jsonl`, frozen from
direct per-speaker audio/alignment/video filename intersections before media
decoding or prediction inspection. Its SHA-256 is
`e897ff02a435a463211f0a157495058adcb5c390bf384437e3e56dc8066da281`.

The fixed task is: given an endpointed GRID utterance and four same-speaker
candidate clips, assign probability to the synchronized clip. The model sees
neither speaker ID, transcript, source code, quartet ID nor target label. The
sample has 64 two-slot quartets and all four query roles per quartet (256
queries), eight speaker groups with 32 queries each, 255 distinct audio assets
and 255 distinct video assets. Each target option position occurs 64 times.

The model, heads, preprocessing and calibration were frozen from the existing
Lombard runtime. No fitting, calibration, threshold/temperature search,
training, checkpoint choice, sealed-audit read or GPU inference occurred. This
is the same fixed validation sample for every condition. GRID speaker groups
are the bootstrap units; identity overlap with the Lombard training sources is
still unknown, and the sample is not a blind audit.

## Source, runtime and integrity

All eight direct per-speaker audio archives, alignment archives and video
metadata tails matched their saved SHA-256 receipts. The corrected selection
contains 255 paired source identities; 223 selected MPG members came from
HTTP byte ranges across seven Zenodo archives, and 32 from the already verified
complete s30 ZIP. Every ranged response's status, `Content-Range`, local and
central ZIP headers, uncompressed size and CRC-32 were checked; all 223 saved
members then passed SHA-256 verification. The audio tar hashes were verified
against the direct-source audit before member extraction. No source media or
weights were added to Git.

Four initial range transfers (1,638,400 bytes in 10.63 seconds) averaged
0.154 MiB/s per member and 0.147 MiB/s pooled, predicting about 10.7 minutes
for the 95,794,223 compressed selected bytes. The complete 223-member transfer
recorded 97,830,912 bytes and 528.06 summed member seconds (0.177 MiB/s pooled;
0.181 MiB/s mean per member). The extra bytes reflect selected members' local
ZIP headers. This is the observed runtime of this network path, not a future
throughput guarantee.

The successful inference used a separate process and output directory:

```powershell
python -m scripts.evaluate_grid_direct_speaker_pilot `
  --output C:\CodexArtifacts\pretrained-reuse-r2\grid-av-direct-fixed-eval-20261028-attempt-02 `
  --worker-batch-size 32
```

Preflight command:

```powershell
python -m scripts.evaluate_grid_direct_speaker_pilot --preflight-only
```

Preflight returned `PASS_PREFLIGHT`: 64 quartets, 256 queries, 255 paired
audio/video assets, 223 range-fetched videos, eight source-hash groups. It did
not use the GPU or read sealed audit. The exact successful attempt took
894.12 seconds (14.90 minutes) on CPU, using eight VideoPrism worker batches
(32, 32, 32, 32, 32, 32, 32, 31 videos). Every worker reported
`PASS_NATIVE_VIDEO_WORKER`; its frozen parameter hash was unchanged before and
after every batch. Peak reported worker RAM was 3.98–4.00 GiB. There was no
cloud use or cost. The RTX 3080 Laptop remained occupied by the unrelated
Orbitune process and was not used or interrupted.

Environment: Intel i7-12700H, Windows x64, Python 3.11.9, PyTorch 2.6.0+cu124
(CPU execution), Transformers 5.6.2, NumPy 2.4.6, PyAV 18.1.0, float32 model
inputs and float64 probability scoring. Pinned checkpoint, processor, source,
calibration and code hashes are in the external evaluation JSON. The source
run code hash was `effe4289545a3e416447dfd6edad5f45a1d8a145f8742d15bc89c85764d0537c`.

The first evaluator attempt is retained separately at
`grid-av-direct-fixed-eval-20261028-attempt-01`. It stopped before producing
predictions after a dimension error exposed a wrong normalizer-key selection
in the new runner. The correction used the checkpoint's pinned 768D video
statistics instead of the 384D audio statistics and added dimension checks.
The successful run used the corrected code and a new attempt directory.

## Fixed-sample results

Multiclass Brier is the sum of squared class-probability errors averaged over
examples. ECE uses ten equal-width top-confidence bins. Missing-modality rows
exercise the runtime's explicit uniform fallback; they are **not** separately
trained or inferred unimodal models.

| Condition | Accuracy | NLL | Brier | ECE | Mean confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Joint matched audio + video | 34.38% | 2.0492 | 0.9109 | 0.3369 | 0.6807 |
| Uniform missing-audio policy | 25.00% | 1.3863 | 0.7500 | 0.0000 | 0.2500 |
| Uniform missing-video policy | 25.00% | 1.3863 | 0.7500 | 0.0000 | 0.2500 |
| Audio from another candidate in the same quartet | 24.61% | 2.6652 | 1.1121 | 0.4505 | 0.6957 |
| Video options rotated against the original IDs | 17.97% | 2.8363 | 1.1756 | 0.5065 | 0.6807 |

The joint classifier exceeds the balanced 25% accuracy reference but has worse
NLL and Brier than the uniform policy: it is much too confident on this sample.
Changing the candidate-video correspondence has a strong negative effect. The
audio mismatch changes accuracy less; its 95% grouped interval nearly touches
zero. This is evidence that the supplied video correspondence affects the
fixed decision rule, not proof that either tower solves open-world AV
reasoning. Because the model's missing-input rule emits a uniform vector, the
two missing-input rows do not isolate the frozen single-modality heads.

Paired whole-speaker bootstrap used 10,000 resamples, seed 1701, percentile
95% intervals, keeping each speaker's 32 query roles together:

| Joint minus condition | Accuracy difference | 95% interval |
| --- | ---: | ---: |
| Uniform missing-audio policy | +9.38 pp | [+3.52, +15.63] pp |
| Uniform missing-video policy | +9.38 pp | [+3.52, +15.63] pp |
| Mismatched audio | +9.77 pp | [+0.39, +19.53] pp |
| Mismatched video | +16.41 pp | [+7.42, +26.17] pp |

Accuracy/NLL/Brier/ECE by speaker (32 queries each):

| Speaker | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| s13 | 28.13% | 2.7469 | 1.1114 | 0.5274 |
| s18 | 31.25% | 2.0401 | 0.9413 | 0.4280 |
| s23 | 50.00% | 1.5277 | 0.6940 | 0.1654 |
| s24 | 43.75% | 2.0667 | 0.8370 | 0.3643 |
| s25 | 37.50% | 1.5867 | 0.7979 | 0.2908 |
| s27 | 21.88% | 2.3926 | 1.0711 | 0.4090 |
| s30 | 25.00% | 2.5119 | 1.0653 | 0.4944 |
| s4 | 37.50% | 1.5213 | 0.7693 | 0.2209 |

Variation across eight speaker clusters and poor calibration make the overall
accuracy unsuitable as a stable quality estimate. The output is a task-specific
four-choice probability, not the project's full typed JSON Decision contract.

## Interpretation and next gate

This is a narrow speech/lip correspondence transfer test. It does not measure
text reasoning, standalone image understanding, general audio/video tasks,
long temporal state, sensor fusion, product-device latency, or the R0
95%/98% target. Source speaker independence against Lombard training remains
unverified, and the eight-speaker development sample cannot close an audit
gate. Do not tune this fixed selection or promote the model.

The result supports continuing Path A diagnosis, but not spending on a new
backbone or shared Lens yet. Before any GRID training, verify the complete
24-speaker train partition's source mapping, actual per-speaker identities, and
cross-source overlap. Investigate the confidence collapse with training-side
feature/readout diagnostics or a separately frozen task-matched source. If an
independent source and split cannot be established, record the task as
inconclusive and move to another admitted task rather than tuning this sample.

## Receipts

- Selection: `e897ff02a435a463211f0a157495058adcb5c390bf384437e3e56dc8066da281`
- Source-selection summary: `cf572ae2e71ccfc1acd939d829b5880d897175422a964e8fe0c5d064d6022412`
- Range-extraction manifest: `4816e81794b60e40b8a0ec8410302864647ab5f230420408fc3f7306c8e40946`
- Evaluation JSON: `C:/CodexArtifacts/pretrained-reuse-r2/grid-av-direct-fixed-eval-20261028-attempt-02/evaluation.json`
- Evaluation JSON SHA-256: `23f49af66fa7ed452d5333e7511249b9ce8a43e8cc2a2bf24d9efc8e0c4a1552`
- Failed attempt: `C:/CodexArtifacts/pretrained-reuse-r2/grid-av-direct-fixed-eval-20261028-attempt-01/`
- Full outputs and media remain outside Git; evaluation predictions and worker
  receipts are preserved in the successful attempt directory.
