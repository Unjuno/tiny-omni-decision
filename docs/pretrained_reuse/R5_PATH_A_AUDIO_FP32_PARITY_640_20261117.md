# Path A Audio FP32 parity on 640 fixed development assets — 2026-11-17

## Scope and method

This is a numeric-runtime comparison only. The frozen Path A bundle was run on
640 previously observed Speech Commands development assets, using four fixed
queries per asset (2,560 decisions total), batch size 8, CPU FP32 versus CUDA
FP32 with CUDA matmul and cuDNN TF32 disabled around the Whisper forward.
No training, checkpoint selection, sealed-audit access, or quality metric was
performed. Targets were not copied into the requests or used in calculations.
The set is narrow: every clip is 0.426625–1.0 seconds; it does not cover longer
speech or establish general cross-device parity.

One preparatory schema inspection of the existing development JSONL printed a
single row containing a target field before the diagnostic script was written.
The run itself subsequently selected only IDs, question text, options and task
type, and its metrics are label-free. This was a limited development-label
exposure; no target was used for model or configuration selection. This is not
sealed-audit data.

## Results

| Measure | CPU FP32 vs CUDA FP32-no-TF32 |
|---|---:|
| Decisions / option probabilities | 2,560 / 10,240 |
| Maximum absolute probability delta | 1.8239021e-5 |
| Mean absolute probability delta | 1.1533988e-7 |
| Median / p95 / p99 absolute delta | 4.8021e-10 / 4.1723e-7 / 2.7956e-6 |
| Decisions with per-decision max delta > 1e-5 | 2 |
| Argmax changes | 0 |
| Strict max-delta gate (1e-5) | FAIL |

Both outliers belong to `speechcmd-validation-8547d97fb2c8376cf13d`, one
`switch_presence` and one `keyword_identity` query. Their per-query maxima are
1.8239021e-5 and 1.5199184e-5. The other 2,558 decisions meet the threshold.
The outliers did not change the chosen option. Thus the prior four-query pass
was not representative enough to establish the strict gate over this broader
fixed sample; the result does not show a decision-quality regression.

CPU inference took 94.54 s; CUDA inference took 20.18 s after load. CUDA peak
allocation/reservation was 462,284,288 / 564,133,888 bytes. Runtime was Python
3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2, on an NVIDIA RTX 3080 Laptop
GPU. This was one run, not a latency benchmark. During post-run inspection the
GPU was at 99% utilization (8,270/16,384 MiB); Orbitune PID 27140 remained
running and was not stopped or modified.

## Reproduction and artifacts

Source commit: `4ed40db73d6ce9b824b4e8a873e78d0b80473639`.
Script: `scripts/diagnose_path_a_audio_validation_precision.py`.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
python scripts/diagnose_path_a_audio_validation_precision.py `
  --bundle-dir C:\CodexArtifacts\pretrained-reuse-r2\path-a-audio-multiquery-runtime-bundle-v2 `
  --reference-jsonl C:\CodexArtifacts\pretrained-reuse-r2\audio-multiquery-v0\validation-multiquery-predictions.jsonl `
  --sample-dir C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-sample-seed17-128-64-retry2 `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\path-a-audio-no-tf32-640-20261117-attempt-01
```

- Input sample-order SHA-256: `20dca814ebf78e19ce0d81d77a2f87fec615d9496c20b7567347347ec9073b70`
- Reference JSONL SHA-256: `e203cb22ba5e18d75b0a4a78afcd475993f556edefedc7cf12aeb5210a5b3932`
- Bundle manifest SHA-256: `574640d8` (prefix only; full hash remains in the run output)
- `parity-diagnostic.json` SHA-256: `ad0d4d4beb2cd5ecb040420ceca9c6dbf33194cd78dfdbde657ef829c7555bfd`
- CPU predictions SHA-256: `b12129647bb12765df07a87418ba890671fc7a27e8142413bda0e05fe130c8fa`
- CUDA predictions SHA-256: `14a7886502fea538bbbe7a08f6f28da9b5b54c9874e3c40d0562df25fdf1641f`
- Audio / Text / readout SHA-256: `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395` / `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433` / `4efa7939a686ad97b38f9eec0b8171e6e62d125ba4eb373e5d83e33ac4df42c7`.

The full result JSON and prediction files remain outside Git under the output
directory above. The model bundle and source corpora were not modified.

## Decision

Keep `auto` as the default. The explicit no-TF32 path preserved every argmax
on this set, but fails the predeclared strict maximum probability tolerance on
two decisions. Do not describe it as passing broad numeric parity. No further
GPU run was started because the separate Orbitune workload occupied the GPU.
A future precision-contract decision needs a preregistered, duration-diverse
label-blind input set and target-device latency measurement; do not tune from
these two outliers alone.
