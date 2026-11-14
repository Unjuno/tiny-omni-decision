# Path A canonical JSONL runtime (2026-11-14)

## Purpose and scope

The frozen Path A routes already existed as separate runtimes and a Python
dispatcher. This change adds a stable JSONL command entry point that composes
those runtimes and emits only the typed supplied-option Decision JSON records.
It does not add weights, train a model, tune a checkpoint, calibrate outputs,
or claim that the modalities complement one another.

Supported tasks are the existing task-scoped routes:

- Text: synthetic Typed Decisions Synth state/question/options.
- Image: CLEVR-4 taxonomy choices from a raw image.
- Audio: four fixed Speech Commands query templates from raw audio; one request
  may return several typed records.
- Video: CLEVRER binary candidate decisions from raw video, with VideoPrism
  frame encoding and the frozen Torch scoring path.

The supplied output contract is:

```json
{"task":"text","payload":{"state":"...","question":"...","options":["A","B"]},"decision":{"request_id":"r-1","decision_type":"choice","option_ids":["a","b"],"option_texts":["A","B"]}}
```

Audio uses a `decisions` array with exactly one typed decision spec per item in
`payload.queries`. The process accepts one JSON request per input line and
emits one JSON record per line; an audio response is an array of records. It
validates option ordering before scoring. Runtime errors go to stderr and
return a nonzero status; stdout remains the JSON response channel.

The CLI checks the assembly manifest's unique parameter count against a
default 200,000,000 limit. It writes the pinned manifest hash, package bytes,
and worker revision/hash/load metadata to a new `--metadata-output` path.
Worker logs and metadata paths are never overwritten.

## Implementation

- `src/tiny_omni_decision/path_a_jsonl.py`: request validation and JSONL loop.
- `tiny-omni-decision-path-a`: installed command entry point.
- `tests/test_path_a_jsonl.py`: typed request parsing, multi-query Audio,
  strict malformed-input handling and JSON output tests.
- Existing worker and dispatcher code is reused without model or scorer edits.

Example invocation after installing the package and its ML dependencies:

```powershell
tiny-omni-decision-path-a `
  --assembly C:\path\to\manifest-verified-runtime-view `
  --videoprism-python C:\path\to\videoprism-venv\Scripts\python.exe `
  --videoprism-source C:\path\to\pinned-videoprism-source `
  --worker-log-dir C:\path\to\new-attempt\worker-logs `
  --metadata-output C:\path\to\new-attempt\runtime-metadata.json `
  --device cuda < requests.jsonl > decisions.jsonl
```

The current v4 assembly directory includes a root `README.md` that is not
listed in its package manifest. Strict verification correctly rejects that
directory because the file set differs. The measured CLI run therefore used
a separate hard-linked, manifest-only view; the source assembly and weights
were read-only. Do not weaken the exact file-set/hash verifier to accept the
unlisted file.

## Measured runtime verification

Source stack: PR #92 integration commit `6216b6dadd66d06981488a2fd900195a2f87ea88`;
JSONL implementation commit `217d9aeac740425e3b87581da45335c5eb626ba2`.
Assembly manifest SHA-256:
`79FBCA05E6062B66FE6FFB8A831D6C635EDFFE361C76EA4CE1574F0736AE9519`.
The final JSONL invocation used the local RTX 3080 Laptop GPU for the Torch
worker and `TFRT_CPU_0` for VideoPrism, with no training or sealed-audit
access.

| Route | Fixed decisions | Typed JSON | Saved-probability parity | Max absolute probability delta |
|---|---:|---|---|---:|
| Video | 1 | Pass | Pass | 1.19e-7 |
| Image | 1 | Pass | Pass | 1.28e-8 |
| Text | 1 | Pass | Pass | 4.47e-7 |
| Audio | 4 | Pass | Pass | 8.38e-9 |

The final CLI command completed in 27.56 s including process startup and all four routes.
The Torch worker loaded MiniLM once (`1` shared instance), reported 55,986,819
unique Torch-worker parameters and 225,261,056 CUDA allocated bytes after
load. VideoPrism reported 114,365,184 encoder parameters and ran on CPU. The
verified assembly manifest records 170,499,716 unique total runtime parameters
and 687,254,772 package bytes excluding its manifest. The parameter gate
passes; the provisional 200 MB package target does not.

The four requests are frozen development examples used previously for parity,
not a new quality sample. They establish route wiring, caller-ID/option-order
preservation, serialization, pinned payload loading and output parity only.
They do not estimate Accuracy, NLL, Brier, ECE, OOD performance, or joint AV
benefit.

## Attempts and corrections

All attempts were preserved under
`C:\CodexArtifacts\pretrained-reuse-r2\path-a-canonical-dispatcher-20261114-attempt-*`
(the path resolves to `D:\CodexArchiveRelocated\CodexArtifacts` on this host).

- Attempt 01 used CPU. Typed JSON passed on all routes and Video/Image/Text
  parity passed, but Audio maximum probability delta was `0.000617743`, above
  the unchanged `1e-5` parity gate. No tolerance was relaxed.
- Attempt 02 reran the fixed dispatcher comparison with CUDA for the Torch
  worker; all four routes passed typed JSON and saved-probability parity.
- Attempts 03–05 exposed invocation issues: source checkout import path was
  missing, the unmodified assembly had an unmanifested README, and a Windows
  junction path was compared before resolution. Each failed attempt remains
  intact. Attempt 06 used `PYTHONPATH=src` and a resolved manifest-only view;
  it ran the new JSONL CLI and passed all four route checks before the code was
  committed. Attempt 07 repeated the CLI run against the committed source
  revision and is the authoritative result above.

Report hashes:

- CPU dispatcher attempt 01: `1c5a5c1dacdf13850ef6078900729b7a7c98773c8ddab652f184732eaf49f36e`.
- CUDA dispatcher attempt 02: `6b29c6de66de3c8e8906268b8b80aa46fdd08a00144335f2da83e6b282f9b191`.
- JSONL CLI verification report, attempt 07: `9c1fbddacfc0e7222429cd2be00c88da3391600b39c8172c6a26b0457ed9f535`.
- JSONL request: `ae200aa3c099a2d8ead6d1dd1890122cf8b093ac1989fe0876f69735b267123e`.
- JSONL output: `0cd3437ecd32310a724f2d982c641c7497ff88bab3205c01f07c79e7aebf675a`.
- Runtime metadata: `8cabab058bc4f5e299cddaae282e5a8ce288e59cdb0124c60d1ffdeb585e5be4`.

The full CPU suite passed **2,912 tests, 11 skipped**; Ruff passed. The same
local manifest/catalog validation commands used by CI passed for model and
dataset manifests and all four candidate catalogs.

## Remaining gates

This implements an executable, typed four-route Path A runtime, not a single
shared multimodal encoder or a joint AV Decision model. The text, audio, image
and video tasks remain narrow and task-specific. R2 still needs a rights-cleared
paired-input task with an independent group split and evidence that both inputs
are needed. The 200 MB package target, task quality requirements, calibration,
OOD and release/device gates remain open. No quantization or product promotion
was performed.
