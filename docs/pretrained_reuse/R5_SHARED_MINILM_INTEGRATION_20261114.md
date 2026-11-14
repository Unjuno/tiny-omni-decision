# Shared MiniLM integration on current Path A stack (2026-11-14)

## Scope

This change carries the already implemented and measured PR #66 shared-MiniLM
startup optimization onto the newer Path A runtime branch without importing
the unrelated commits between their common ancestor and PR #66. PR #66 and its
artifacts remain unchanged. The integration changes when the existing Audio,
Text and CLEVRER Video routes construct the shared MiniLM; it does not change
weights, readouts, route semantics or the Decision contract.

- Integration branch: `codex/path-a-shared-minilm-integration-20261114`.
- Parent Path A head: `73bc8470ed02a65ec4f4f81ec3ee41e2ab68d41c`.
- Implementation source commit: `b7653680f235e276c20fd3c4cb4608e72133ae70`.
- Measurement-documentation source commit: `ca70e572b14e3a00fbd800b5f71d3f7df5d5b7e2`.
- Integrated implementation commit: `14eb8051` (with cherry-pick provenance).
- Integrated documentation commit: `aabc5eeb` (with cherry-pick provenance).

The worker loads MiniLM once from the hash-pinned local component, freezes it,
then injects the same encoder and tokenizer objects into each route. Each route
continues to check its local MiniLM and readout hashes. Standalone runtime
constructors retain their original self-loading behavior, and supplying only
one of encoder/tokenizer fails closed. The worker still exposes only its
existing `text`, `audio`, and `video_score` routes.

## Verification

Focused CPU tests used the existing Python 3.11.9 / PyTorch 2.6.0+cu124 /
Transformers 5.6.2 environment; no dependency was installed or upgraded:

```powershell
python -m pytest -q tests/test_path_a_shared_minilm_loading.py tests/test_audio_runtime.py tests/test_text_runtime.py tests/test_video_runtime.py tests/test_runtime_dispatch.py
```

Result: **29 passed**. Ruff passed on the five changed Python files, and
`git diff --check` passed.

A separate CPU process then loaded the existing clean v4 assembly read-only
and ran one synthetic request through each current route: a typed text choice,
a generated tone WAV through the audio route, and a fixed zero video-feature
vector through the video scorer. The output contained one ready record and
three route responses. All three responses had two finite probabilities that
sum to 1 within floating-point tolerance; the worker reported one MiniLM load,
one shared instance, 55,986,819 unique resident parameters and clean exit code
0. No dataset quality metric was calculated.

| Measurement | Result |
|---|---:|
| Device | CPU |
| MiniLM loads / instances | 1 / 1 |
| Unique resident parameters | 55,986,819 |
| One-run model load time | 2.4982 s |
| RSS before load | 527,626,240 bytes |
| RSS after load | 557,592,576 bytes |
| Peak working set after load and requests | 703,311,872 bytes |
| CUDA initialized / used | No / 0 |

The one-run load and working-set values are descriptive only; no before/after
performance comparison was repeated. The prior PR #66 CPU/GPU measurements
remain attributed to that earlier stack in
[`R5_SHARED_MINILM_STARTUP_20261111.md`](R5_SHARED_MINILM_STARTUP_20261111.md).

External attempt directory:
`D:\CodexArtifacts\pretrained-reuse-r2\path-a-shared-minilm-integration-20261114-attempt-01\`.

- Request JSONL SHA-256: `D7B80FEED4AFFDB7826540CD11308CBEEB827C5719EA36D4C014E5E3C5688E80`.
- Response JSONL SHA-256: `6592336756528BAC34CEC0AC07EB6AA4896CA2798F171D4B16E6CFF70625B634`.
- Read-only assembly manifest SHA-256: `79FBCA05E6062B66FE6FFB8A831D6C635EDFFE361C76EA4CE1574F0736AE9519`.

## Limits

This integration does not change total parameters, prove broader quality,
establish multimodal information fusion, close R2 M-2, or meet the product
package-size goal. The worker's Video route still scores pre-extracted video
features. No training, validation selection, sealed audit, new media, GPU
execution, or model artifact modification occurred. Existing Path A runtime
outputs and the PR #66 performance artifacts were read-only.
