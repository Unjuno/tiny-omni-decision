# Path A shared MiniLM startup measurement — 2026-11-11

## Scope and source

This bounded R3 runtime-efficiency check uses the clean modular Path A assembly `path-a-modular-runtime-assembly-v4-clean-copy-20261026-01` and source commit `bc1754097ecb584b1d26a8efb1b744c1e441747f` (the Path A runtime branch underlying PR #63). The pre-change worker constructed Audio, Text, and CLEVRER Video runtimes independently. Each initialized the same pinned MiniLM encoder/tokenizer; the worker then consolidated references, leaving only one unique parameter set resident. Thus resident parameter accounting was already deduplicated, but initialization work and transient memory were not.

The change adds optional encoder/tokenizer injection to those three runtime constructors. Standalone callers retain their existing self-loading behavior. The Torch worker now loads the pinned MiniLM once, freezes it, injects the same objects into the three routes, and reports one shared load/instance. Supplying only one of encoder/tokenizer fails closed. No model weights, task readouts, task data, decision contract, or training code changed.

## Fixed CPU smoke comparison

Both arms used the same assembly, pinned revisions/checkpoint hashes, Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2, FP32, CPU, and equivalent worker-ready path. The pre-change worker script was snapshotted from the source commit into the attempt directory; the new worker used the change in this branch. Existing artifacts were read-only.

| Measure | Before: attempt-05-baseline | Shared load: attempt-03 | Difference |
|---|---:|---:|---:|
| MiniLM initialization count | 3 | 1 | −2 loads |
| Resident unique parameters | 55,986,819 | 55,986,819 | 0 |
| RSS before loading | 535,711,744 B | 535,138,304 B | — |
| RSS after loading | 572,051,456 B | 560,361,472 B | −11,689,984 B |
| Peak working set after loading | 585,834,496 B | 560,361,472 B | −25,473,024 B (24.29 MiB) |
| Reported model load time | 0.6063 s | 0.4545 s | −0.1517 s |

These are single CPU runs per arm; allocator/OS variation is not controlled and these deltas are not a performance guarantee. The reliable structural result is three loads reduced to one while preserving one resident model.

## Bounded RTX 3080 startup comparison

After confirming `nvidia-smi` reported 0% GPU use and 0 MiB allocated before each arm, the same baseline worker snapshot and shared-load worker were run sequentially in separate processes on the local NVIDIA RTX 3080 Laptop GPU (16 GiB). Each exited cleanly after reporting ready. The fixed assembly, runtime venv, pinned checkpoints, FP32 precision, and CUDA 12.4 / Torch 2.6.0+cu124 environment were unchanged. No training or input dataset was used.

Exact worker commands (both used the same assembly and environment):

```text
D:\CodexArtifacts\pretrained-reuse-r2\path-a-runtime-py311-cu124-20261110-attempt-01\venv\Scripts\python.exe D:\CodexArtifacts\pretrained-reuse-r2\path-a-shared-minilm-load-20261111-attempt-05-baseline\path_a_torch_worker.py --assembly C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4-clean-copy-20261026-01 --device cuda
D:\CodexArtifacts\pretrained-reuse-r2\path-a-runtime-py311-cu124-20261110-attempt-01\venv\Scripts\python.exe D:\CodexWorktrees\path-a-shared-minilm-load-20261111\scripts\path_a_torch_worker.py --assembly C:\CodexArtifacts\pretrained-reuse-r2\path-a-modular-runtime-assembly-v4-clean-copy-20261026-01 --device cuda
```

| Measure | Before: baseline | Shared load | Difference |
|---|---:|---:|---:|
| MiniLM load count | 3 | 1 | −2 |
| Resident unique parameters | 55,986,819 | 55,986,819 | 0 |
| Startup peak CUDA allocated | 364,396,032 B | 225,261,056 B | −139,134,976 B (132.69 MiB) |
| Startup peak CUDA reserved | 396,361,728 B | 236,978,176 B | −159,383,552 B (152 MiB) |
| CUDA allocated after load | 225,261,056 B | 225,261,056 B | 0 |
| CUDA reserved after load | 308,281,344 B | 236,978,176 B | −68 MiB |
| Reported model load time | 1.2735 s | 0.6380 s | −0.6355 s |
| Process wall time | 19.04 s | 4.69 s | −14.35 s |

The peak allocation difference matches the two avoided FP32 MiniLM parameter copies (about 139.1 MB decimal). The ordering was baseline first and treatment second, with one run per arm; CUDA initialization/cache/order effects can affect load and process wall time, so those timing differences are indicative only. The persistent post-load allocated model footprint was identical. This does not establish a repeated latency distribution or product performance.

## Functional parity

The three-route CPU canary returned successful responses. Against preserved pre-change canaries, probabilities were exactly equal (max absolute delta 0.0) for Text (`smoke05-text`), Audio (`smoke05-audio`, both queries), and Video (`smoke06-video`). This is execution and parity coverage only; the examples are not a quality benchmark and no accuracy claim follows.

Artifacts:

- Baseline: `D:\CodexArtifacts\pretrained-reuse-r2\path-a-shared-minilm-load-20261111-attempt-05-baseline` — metadata SHA-256 `7b6a552484c3544368462c75f0994e7970acec3786b116cfaed0175de1e7bc84`; exit 0, clean shutdown.
- Treatment: `D:\CodexArtifacts\pretrained-reuse-r2\path-a-shared-minilm-load-20261111-attempt-03` — run metadata SHA-256 `8ae9d861d1e224e9cbb284d862a252a2df91902ab3bff5ffcbf7c23d0c140391`; requests SHA-256 `435d5fb0ec14725dab43f941692fd84c426afae2c0b74b1c5fa32fca2531ddaa`; responses SHA-256 `7e8f78185593ac42122922583eec07a4332747291c64dbc7074595f8b43279bd`; exit 0.
- CUDA comparison: `D:\CodexArtifacts\pretrained-reuse-r2\path-a-shared-minilm-load-20261111-attempt-06-cuda` — run metadata SHA-256 `f39c92f6719a8fa9e64b8a92b62e4b3fada419cb09c9cfe017e2eb4de3af415d`; baseline stderr SHA-256 `137df3dd6f8e6416a3109d6646fd3a431e9197bf908ce5714911669c11b000dc`; shared stderr SHA-256 `ede24af90291a68e46072e9fa65d30a70a151e1d761d1b75c1981a17757191e4`; both exit 0 with clean shutdown.
- Two earlier failed harness attempts and the first abnormal baseline wrapper attempt remain preserved in their separate attempt directories; they were not overwritten.

No training, validation, sealed audit, external dataset, or model-quality evaluation was used. No existing assembly, checkpoint, corpus, or Teacher artifact was modified. Both GPU processes were our bounded startup comparison; they exited cleanly and the RTX 3080 returned to 0% / 0 MiB afterward. No cloud compute or spend was used.

## Verification and limits

`ruff check` passed on all changed Python files. Focused CPU suite: 40 passed. Added regression coverage verifies one loader call per MiniLM model/tokenizer, shared object identity across routes, frozen parameters, unique parameter counting, and rejection of partial injection.

This is a small runtime-efficiency improvement for an existing Path A worker. It does not demonstrate cross-modal information fusion, source/task admission, or R2 task quality and does not satisfy the broader R5 goal by itself. Further work should prioritize a source/task gate and quality evaluation on a valid, non-contaminated task before architecture expansion.
