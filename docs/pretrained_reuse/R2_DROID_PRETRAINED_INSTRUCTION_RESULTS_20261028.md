# DROID frozen pretrained-instruction probe: result

The [fixed protocol](R2_DROID_PRETRAINED_INSTRUCTION_PROTOCOL_20261028.md)
ran on CPU over the same 275 train recordings, 268 connected components and
five folds as the earlier word-hash comparison. The 150 nonempty instructions
all fit within the 128-token limit. The local checkpoint and tokenizer matched
their prior receipt; an independent two-input CPU smoke confirmed 384-value
text output and the missing-input flag.

| Inputs | Rows | Accuracy | NLL | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|
| Sensor | 275 | 36.36% | 2.6591 | 0.9121 | 0.3063 |
| Sensor + frozen MiniLM CLS | 275 | 34.18% | 2.9588 | 0.9348 | 0.3627 |
| Sensor, text present | 150 | 37.33% | 2.7794 | 0.9091 | 0.3081 |
| Sensor + frozen MiniLM CLS, text present | 150 | 37.33% | 3.1602 | 0.9209 | 0.3533 |

The sensor baseline exactly reproduces the earlier probe's aggregate results.
The added embedding did not improve this fixed readout's overall accuracy and
made its probability metrics worse. The pretrained feature is from a retrieval
cross-encoder and was not fine-tuned for robot commands; the input dimensions
and trainable head size also differ. This result neither establishes the value
of instruction conditioning generally nor evaluates video/audio fusion. This
small train-only development study is not an independent validation benchmark.
The 22,713,601-parameter text checkpoint must be added to other runtime encoder
counts: this experiment does not prove a complete four-modality model fits the
200M prequantization limit.

Artifacts:
`C:/CodexArtifacts/pretrained-reuse-r2/droid-pretrained-instruction-probe-20261028-attempt-01/`;
`result.json` SHA-256
`18f83eeaeaef497557f4cb7ae31343cc65835a5f7c17afe989aaad9dbead78aa`.
The result contains
per-fold metrics, hashes and OOF predictions; no instruction text is stored.
Ruff passed on the script, and the matching prior probe's three CPU tests pass.
