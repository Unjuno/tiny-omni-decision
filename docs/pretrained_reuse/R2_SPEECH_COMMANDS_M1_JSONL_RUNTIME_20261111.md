# Speech Commands M1 JSONL runtime integration — 2026-11-11

## Result

The fixed English Speech Commands M1 readout now runs through the task-scoped
Path A typed JSONL interface. This is a runtime integration result only. It is
not a new evaluation, training run, model-selection event, or product-quality
claim.

## Frozen task and provenance

- Task ID: `path-a-audio-speech-commands-m1-v0`.
- Task: choose among all ten fixed labels `yes`, `no`, `up`, `down`, `left`,
  `right`, `on`, `off`, `stop`, `go`.
- Source: `google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da`,
  official TRAIN shards; publisher declares CC BY 4.0.
- Source train manifest SHA-256:
  `dafba4f510d1ca49fe89d58ac9a4ea0f4268081b8689cbafeb25b46739c2df6e`.
- Frozen encoder: `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af`,
  37,760,640 parameters, SHA-256
  `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`.
- Frozen M1 readout: 3,850 parameters, 16,950 bytes, SHA-256
  `cf5f5b2e552c3adc82afe86650d1c765f5fb2d3e6f3127b1406ba35c3b5ebfa8`.
- Temperature: `0.47748440347754983`, fitted on the pre-existing speaker-held-out
  TRAIN calibration subset. Runtime code hash-checks the readout and does not
  refit or change calibration.
- The already-observed development result is 95.1% Accuracy, 0.1655 NLL,
  0.0654 Brier, and 0.0225 ECE-15 on 1,000 fresh-speaker development rows.
  That result belongs to the separate M1 fit/evaluation record; this integration
  did not load or score those rows. It is task-specific, development-only
  evidence, not a general audio result.

## Runtime implementation

The route reuses the Path A worker's already-loaded frozen Whisper encoder and
its exact 384-wide mean-pooled feature contract. It checks the input WAV SHA-256,
hash-checks and loads the frozen readout, applies the frozen temperature, and
maps the canonical ten-class probabilities back to the caller's option order.
The JSONL contract rejects missing/invalid media hashes, partial label sets,
duplicate/mismatched labels, and non-choice decision types before dispatch.
The JSONL output now includes loaded-worker revisions, unique parameters, and
memory counters while omitting the local worker process ID.

This is not ASR, open-vocabulary speech understanding, or joint audio-video
reasoning. It covers one narrow command-recognition task only.

## Implementation and validation

- Base branch: the four-route Path A integration head from PR #57,
  `519ba902e9cc61b8c00d9f3933fae23cd9fe9b15`.
- Runtime integration branch: `codex/r2-speech-commands-m1-runtime-20261110`.
- M1 route implementation commit:
  `c2bfce9be8521effd538037a5eb158ff4a966a8d` (`feat: add fresh-speaker Speech
  Commands JSONL route`). Runtime startup-metadata output and its regression test
  are in `ab8db807cc788bd73490d296c3b672e338de1927` (`docs: record Speech
  Commands M1 Path A runtime`). The
  smoke ran on the exact file contents later committed there; that commit also
  added the regression test and documentation.
- SHA-256 of implementation files at the smoke revision:

  | File | SHA-256 |
  |---|---|
  | `src/tiny_omni_decision/speech_commands_m1_contract.py` | `d36559b6be5fa2f5a11e89405e89ff3bf122b0d0f007b3c06b7317434d0d461d` |
  | `src/tiny_omni_decision/speech_commands_m1_runtime.py` | `2bd42c7fa74228023635b1aaa882cb6217db30b8ded2d41dd254e86275964f10` |
  | `src/tiny_omni_decision/runtime_dispatch.py` | `1e900e9232d83cff8c4a732e3d4a55db823b683b6fdfc5b4657ceac4a93d736d` |
  | `src/tiny_omni_decision/path_a_runtime_cli.py` | `3460bc07f28759c60da70a8253a985378dcebd0f3e8be6d522e4ec5200df68b2` |
  | `src/tiny_omni_decision/cli.py` | `1f25e4811ffb8997c5b008ddd1c6e8b1b87218755cf58c131c4dcb76e1699e20` |
  | `scripts/path_a_torch_worker.py` | `d62d039666e55c3008f28cef41256ad05f3a5c8f6e26ce10f99aac3071cd9785` |
- Focused verification after rebasing onto the four-route head: 23 tests passed
  across the M1 contract/runtime, dispatcher, JSONL contract, and VideoMAEv2
  round-trip modules; Ruff checks and format checks passed. CLI help lists the
  M1 readout option alongside the existing VideoMAE options.
- Offline model loading was enforced (`HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`). No model or data download occurred.

### Local CUDA JSONL smoke

- Attempt directory:
  `D:\CodexArtifacts\pretrained-reuse-r2\speech-commands-m1-patha-runtime-20261110-attempt-03`.
- Run metadata SHA-256:
  `682fa4cbdc1d6569b10edc3829e2b6304665a9187b3cae6dafbb20c3e8a20bcf`.
- Request used train record `scm1-007c9454983af89b4804`, target `right`, from
  `speech-commands-fresh-m1-media-20261110-attempt-04\train.jsonl`; media SHA-256
  `606ce34b946dfef7ef1c229fb60dc3b47a324db960631df5a560c74ff733b4b4`.
- Result: valid JSON with ten finite probabilities summing to one, caller's
  shuffled option IDs/texts preserved, and `right` selected with confidence
  `0.99819934`. Since this is a train example, do not interpret it as accuracy.
- Hardware/runtime: NVIDIA GeForce RTX 3080 Laptop GPU, Python 3.11.9,
  PyTorch 2.6.0+cu124, Transformers 5.6.2, CUDA 12.4.
- Worker: 55,990,669 unique resident Torch parameters, including the M1 readout;
  model load 0.882 s; startup peak CUDA allocated 364,411,904 bytes; after load
  225,276,928 bytes; after request 233,796,608 bytes. Peak working set after
  model load was 817,012,736 bytes.
- Latency: outer one-line CLI wall `10.319 s`; dispatcher `7.345 s`; worker
  startup `6.970 s`; worker inference `0.356 s`. This is one cold process and
  one short, already-materialized input, not a latency distribution.
- The existing four-component assembly manifest SHA-256 is
  `79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`.
  No assembly/model file was modified. The frozen source readout was copied to
  the new D: attempt directory; the original remains unchanged.

## Size accounting and limits

The four-component v4 assembly is 170,499,716 unique parameters and
687,254,772 bytes. The separate VideoMAE image route adds 22,205,968 parameters
and 44,547,594 bytes. Adding the M1 readout gives a simple union of 192,709,534
parameters and 731,819,316 bytes before any compression. Thus the provisional
100–200M parameter goal is within this simple union, but the provisional
200 MB package goal is not. These counts do not establish a jointly loaded,
quantized, mobile, or release-ready package.

The existing M1 development validation was not reloaded or scored in this
runtime work; no sealed audit was opened. No training, checkpoint selection,
quantization, model promotion, or paid compute occurred. A working path now
exists for the narrow M1 task, while R2 joint-modality quality, broader audio
quality, package-size reduction, and R0–R8 release gates remain open.
