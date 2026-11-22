# Path A Audio numeric parity on LibriSpeech — 2026-11-17

## Scope

Compared the frozen Path A Audio+Text runtime on the fixed, previously observed
LibriSpeech dev-clean input manifest. This is a numeric reproducibility test,
not another quality evaluation: no target index/text was accessed by the new
runner, no Accuracy/NLL/Brier/ECE was computed, and no training, calibration,
model selection, or sealed-audit access occurred. The historical manifest
contains target columns, but the runner projects each row to the audio path,
media hash, duration, ID, and the two already-fixed option lists; saved
predictions contain only ID, media hash, condition, and probabilities.

The input conditions are the existing fixed random-distractor and
phoneme/lexical/length-matched hard-distractor options. Both conditions were
run in the same sample order, batch size 8, against CPU FP32 and CUDA
FP32-no-TF32. The strict maximum probability delta criterion remains 1e-5.

## Results

| Condition | Decisions | Max delta | Mean delta | p50 / p95 / p99 per-decision max delta | Over 1e-5 | Argmax changes | Strict gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Random distractors | 1,938 | 1.5676e-5 | 7.7323e-7 | 1.1325e-6 / 4.2319e-6 / 6.0797e-6 | 3 | 0 | Fail |
| Hard distractors | 1,938 | 1.4454e-5 | 7.7124e-7 | 1.1772e-6 / 4.0531e-6 / 5.6624e-6 | 1 | 0 | Fail |

There were four threshold exceedances across 3,876 decisions, each on a
2.905–7.305 second clip. None changed the selected option. The percentile
columns are order statistics over each decision's maximum four-option
probability delta (sorted index `floor(p * (n - 1))`). These are numerical
parity observations only; they do not say anything about correctness or task
quality. This broader duration range confirms that the four-query/640-asset
Speech Commands pass did not establish the unchanged strict parity gate.

| Runtime measure | CPU FP32 | CUDA FP32-no-TF32 |
| --- | ---: | ---: |
| Model load | 0.377 s | 0.548 s |
| Inference incl. FLAC decode/PCM conversion | 439.80 s | 198.10 s |
| Peak allocated / reserved CUDA memory | n/a | 463,009,280 / 568,328,192 bytes |

This single timing comparison is not a benchmark or service-level latency
claim. Hardware was the local RTX 3080 Laptop GPU (16 GiB). Environment:
Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2, PyAV 18.1.0,
NumPy 2.4.6. No environment package was changed. The existing Orbitune process
was not stopped or modified.

## Frozen input and model identity

- LibriSpeech dev-clean example-manifest SHA-256:
  `1f17d86a15faecc3ba3e13cd406dfd0119ef17850d1a7206956a70f4eb801923`.
- Ordered sample-ID SHA-256:
  `709687d926f46907a3c5d447d8921b51cb8f1188163f25fd777b11dfae34384c`.
- Unique assets: 1,938; all file hashes matched the frozen example manifest.
- Duration range: 1.56–24.695 s; <=5 s: 913, <=10 s: 753, <=20 s: 266,
  >20 s: 6.
- Runtime bundle manifest SHA-256:
  `574640d88d6ddbbdb2d12f2466a248ce13c8d9eb3ac542d099b2f08981b690f4`.
- Audio weights SHA-256:
  `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`.
- Text weights SHA-256:
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- Readout SHA-256:
  `4efa7939a686ad97b38f9eec0b8171e6e62d125ba4eb373e5d83e33ac4df42c7`.

## Reproduction and artifacts

Source commit: `711514454c252257ca242e0e8afd2bf88cc4f3a5`.
Runner: `scripts/diagnose_path_a_librispeech_precision.py`, SHA-256
`a52d25b989e3d475327658d66b13c399c070340dc4df20c1f8136b6f5d858376`.
The run used a new output directory and the following command:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
python scripts/diagnose_path_a_librispeech_precision.py `
  --example-manifest C:\CodexArtifacts\pretrained-reuse-r2\librispeech-audio-hard-negatives-v0\evaluation-examples.jsonl `
  --source-root C:\CodexArtifacts\pretrained-reuse-r2\librispeech-dev-clean-hardneg-v0\source `
  --bundle-dir C:\CodexArtifacts\pretrained-reuse-r2\path-a-audio-multiquery-runtime-bundle-v2 `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\path-a-audio-librispeech-fp32-precision-20261117-attempt-01 `
  --batch-size 8
```

Artifacts remain outside Git at that output path:

- `precision-parity.json` SHA-256:
  `9a6b63886890a06b325e43b55812ec39077870a7b78d87b19c0718e80ea4ef0a`.
- CPU random / hard prediction SHA-256:
  `3ebc0664b8dcd2d863c0e9b0b20100c4d2cc1185cbb145807e3e981b25fc11d9` / 
  `524a9d5811bf31e05d77603afaf29bd2123e18ff5fc0c809425f190e3eeb9ac8`.
- CUDA random / hard prediction SHA-256:
  `c8c6ce940aa137d6bbec4e39d546c51a21dbcf8dcf5716b843d61218c50a671` /
  `2ba0b35d26512a0456f6966b5b706b9b674c697569d19b84312a2fbfd016e521`.

Outlier IDs and per-decision deltas, without transcripts or targets:

- Random: `2412-153947-0001` (1.5676e-5), `3170-137482-0014`
  (1.1861e-5), `1272-141231-0014` (1.0073e-5).
- Hard: `6241-66616-0020` (1.4454e-5).

## Decision

Keep the default precision policy unchanged and keep the strict parity gate
open. The no-TF32 mode preserves argmax on all these fixed choices, but fails
the probability tolerance in both distractor conditions. Do not tune from the
four outlier examples. This result is specific to a previously observed
English audiobook development corpus and the pinned local software stack; it
does not establish a general Audio contract or broad speech quality.
