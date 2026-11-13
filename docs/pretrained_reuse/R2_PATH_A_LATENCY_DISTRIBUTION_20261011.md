# Path A four-modality latency distribution — 2026-10-11

## Scope

This extends the one-example dispatcher smoke to eight distinct saved
development-validation assets per modality. For each asset, the runner
measured a first request and a second request that reused the same immutable
input feature where the runtime supports caching. It checked every output
against the saved prediction. This is a runtime/parity measurement, not a new
quality evaluation or a blind test. No training or sealed-audit access occurred.

The second request asks another question about the same Text state, another
taxonomy on the same Image, another candidate on the same Video scene, and
repeats the same four-query Audio request. Selection follows saved validation
order and requires distinct state/media identities for the eight first
requests. Source IDs and file hashes for every pair are in the machine-readable
report.

## Reproduction record

- Branch: `docs/pretrained-reuse-roadmap-20261010`.
- Source commit: `29ce4420996fbc08616bbbe753d9015043dcbd92`.
- Runner SHA-256: `5e9211dfd875f673e61766dea8561f9b190cd77f3b6340f97af59417e9849a93`.
- Package manifest SHA-256:
  `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`.
- Result directory:
  `C:/CodexArtifacts/pretrained-reuse-r2/path-a-latency-distribution-v0-attempt-01/`.
- Result JSON SHA-256:
  `21294547a64396acf406e812d601a192a470dd1b834d447bc49116a3a99c4e9e`.
- Command:

  ```powershell
  python scripts/benchmark_path_a_latency_distribution.py `
    --assembly 'C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4' `
    --videoprism-python 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe' `
    --videoprism-source 'C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source' `
    --artifact-root 'C:\CodexArtifacts\pretrained-reuse-r2' `
    --output-dir 'C:\CodexArtifacts\pretrained-reuse-r2\path-a-latency-distribution-v0-attempt-01' `
    --samples-per-modality 8 --device cuda
  ```
- The package was exposed through a manifest-only hard-link view because the
  existing assembly directory contains generated, unmanifested bytecode. The
  runner kept strict package verification and left the source assembly intact.
- Before the run, the local RTX 3080 Laptop GPU reported 0 MiB used and 0%
  utilization. The only Python processes were two Django development servers.
  No training job was stopped or changed.

## Measured results

All 32 first requests and all 32 same-asset warm requests matched the saved
prediction index, with probability vectors within the runner's `1e-5`
absolute-tolerance gate. The `requests_with_exact_saved_prediction_parity`
counter in the JSON refers to that combined index-and-tolerance gate; it does
not mean bitwise-identical probability floats.

Times below are batch-one dispatcher wall times. P90 uses nearest-rank on eight
assets, so it is the maximum observation for each eight-item distribution.
The Video all-first-request P90 includes service startup; the steady-service
column excludes that single first call.

| Modality | Distinct assets | First request median | First request p90 | First request steady median / p90 | Same-asset warm median / p90 | Warm encoder cache hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Video | 8 scenes | 1.212 s | 16.779 s | 1.206 / 1.237 s | 10.89 / 13.73 ms | 8/8 |
| Image | 8 images | 1.015 s | 1.208 s | 1.015 / 1.208 s | 6.12 / 6.26 ms | 8/8 |
| Text | 8 states | 19.81 ms | 31.29 ms | 19.81 / 31.29 ms | 16.98 / 29.93 ms | n/a |
| Audio | 8 clips, four queries each | 22.78 ms | 222.00 ms | 22.78 / 222.00 ms | 21.88 / 25.30 ms | n/a |

The first Video request took 16.779 seconds because it included startup of both
runtime workers: Torch `6.412 s` and VideoPrism `3.838 s`. Its VideoPrism call
took `6.290 s` including first-use initialization. The remaining seven Video
asset encodes had a median of about `1.21 s`. VideoPrism ran on CPU (`TFRT_CPU_0`);
the task scorer ran on CUDA. The first VideoPrism Image encoding per image had
a median of about `1.01 s`; cached same-image taxonomy requests took a median
of about `6.1 ms` end to end.

Runtime and package measurements:

- Host: Windows 10; Torch worker Python 3.11.9; VideoPrism worker Python
  3.11.16.
- Torch worker: PyTorch `2.6.0+cu124`, Transformers `5.6.2`, RTX 3080 Laptop;
  55,986,819 unique resident parameters after shared MiniLM consolidation.
- VideoPrism worker: JAX `0.6.2`, TensorFlow `2.18.1`, NumPy `1.26.4`, CPU;
  114,365,184 encoder parameters.
- Combined package: 170,499,716 parameters and 687,254,772 bytes. The
  provisional parameter target is met; the provisional 200 MB package target
  is not.
- Combined worker RSS after loading was about 1.513 GB. CUDA allocation after
  Torch load was about 225 MB; Torch startup peak before shared MiniLM
  consolidation was about 364 MB. VideoPrism peak working set after video
  inference was about 4.16 GB.
- Sequential wall time for the 64 measured requests was 37.58 seconds,
  excluding the report write. No cloud compute or package upgrades were used.

## Interpretation and remaining work

This establishes repeatable batch-one runtime parity across a small set of
distinct assets, plus feature reuse for repeated Video/Image observations. It
does not establish service-level latency tails: eight assets are too few for
that, the first XLA call is an initialization outlier, and requests were
sequential. It does not establish task quality, concurrency, power use,
real-time performance, joint audio-video reasoning, or target-device fitness.
Text and Audio have no encoder-side feature cache in this dispatcher.

The runtime profiles remain separate and the 687 MB package remains above the
provisional size target. R2 still needs task-matched quality gates and broader
approved task coverage. Do not start training or select a product candidate
from these latency and replay measurements.

The prior one-example report and all source models, corpora, predictions,
assemblies and validation files remain unchanged. The new result lives in a
separate output directory; no media or model weights were added to Git.
