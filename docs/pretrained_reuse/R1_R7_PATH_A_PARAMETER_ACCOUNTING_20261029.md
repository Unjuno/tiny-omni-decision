# Path A parameter accounting reconciliation — 2026-10-29

## Finding

The native common Path A fit and the v4 modular runtime package are different
model inventories. Their parameter counts are both valid for their own
artifacts, but the 170.5M modular package is **not** the packaged form of the
196.3M native common fit. Do not transfer runtime or quality claims between
them.

## Native common Path A fit

The fixed fit receipt is
`C:/CodexArtifacts/pretrained-reuse-r2/native-common-baseline-fit-01/result.json`
(SHA-256 `aaa1fe96a4be85a71ce146981c6b2e8b58e1308d91fc110e1ece6804b4d8048c`).
It records source commit `72c1eb47dde0c81e503d3a4d783aa5528660186e`, effective
config SHA-256 `45954a7240ee8da32e4b5b5dc3283994c40c4849451f27ff4fce13d3cc118f75`,
889,345 trainable readout parameters, and 196,254,874 total pre-quantization
parameters. The selected step-1,792 readout is
`best.pt`, SHA-256
`3eed6558731de60da82e75e38e599699f29663dd601f942505fec26a48b8dc1d`.

The frozen representation inventory is independently supported by the
feature-cache configs/results and live checkpoint hashes:

| Input path | Exact model/checkpoint | Parameters | Weight artifact bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| Text | `microsoft/deberta-v3-xsmall@262ae02f29173eec1c250f90804dc7edc677dcff` | 70,830,722 | 141,685,964 | `56b89067a4e60362f498f39b7067020911618fad37d64fa23a5f959a04fd4080` |
| Audio speech | `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af`, **encoder-only runtime artifact** | 8,208,384 | 32,839,984 | `d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8` |
| Audio events | EfficientAT DyMN-0.4, source revision `a425fdce92572e602a1d5634799bd9f1f2efa806` | 1,961,239 | 8,086,973 | `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e` |
| Image and video | `google/videoprism-base-f16r288@be719a406d563b66f0ac969e7c94bab8e997c81a`, shared once | 114,365,184 | 457,475,296 | `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29` |

The four frozen entries sum to 195,365,529 parameters. Adding the common
readout gives exactly 196,254,874, within the hard 200M pre-quantization
parameter limit by 3,745,126. The Whisper entry is only the encoder used by the
native feature path; it must not be replaced in this sum by the 37,760,640
parameters of the separate full Whisper-tiny package model. DeBERTa's pinned
model card declares MIT; the other candidate rights and runtime findings stay
as recorded in [`R1_AUDIT.md`](R1_AUDIT.md).

## Modular runtime assembly v4

The strict-verified manifest-only copy has manifest SHA-256
`79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519`. It
records 170,499,716 unique runtime parameters, 319,644,548 component-parameter
sum (shared tensors are referenced by multiple tasks), and 687,254,772 package
bytes excluding the root manifest. `package_target_met` is false against the
provisional 200,000,000-byte storage target.

Its actual unique model inventory is full Whisper-tiny (37,760,640),
MiniLM (17,389,824), VideoPrism (114,365,184), and task-specific readouts.
Audio, Image, Text and Video are separate task components; MiniLM and
VideoPrism are content-shared where their hashes match. This differs from the
native common fit's DeBERTa text model and encoder-only Whisper plus EfficientAT
audio path. It is not a single common readout for the four modality suite.

## Reproducibility checks and limits

On 2026-10-29 the DeBERTa, Whisper-encoder, EfficientAT and VideoPrism files
were re-hashed locally; all four matched their pinned SHA-256 values above.
The native fit receipt, selected checkpoint, native corpus result, and modular
package manifest also matched the recorded hashes. Parameter arithmetic was
recomputed from the pinned cache receipts and the native fit receipt. During
that hash-only audit, no model was loaded and no fit was run. The later CPU
smoke below loaded the existing checkpoint without modifying it; no sealed
audit was accessed.

This resolves the count discrepancy; it does **not** close the product gate.
The native fit's four-modality macro Accuracy was 30.58%, the minimum task
Accuracy was 0%, and the modular package exceeds the provisional byte target.
The trained common model has not been shown to run as the v4 package. Preserve
both artifacts and keep runtime-package parity, end-to-end modality quality,
independent evaluation, and target-device profiling open.

## Cached-feature CPU inference smoke (2026-10-29)

The selected native common checkpoint was loaded on CPU and run against the
first existing validation example from each modality using the same frozen
feature stores as Path A. The run used no GPU, did not train, did not load the
sealed audit, and computed no quality metrics. All four outputs had finite
probabilities summing to 1; model inference took 0.1004 seconds after feature
store setup. This verifies only that the trained readout can consume the
cached native features and produce candidate probabilities. It is not a
quality estimate or a runtime-package parity result.

The external receipt is
`C:/CodexArtifacts/pretrained-reuse-r2/native-common-cpu-inference-smoke-01/report.json`,
SHA-256 `7c9960f96240dbaae6bd4cfc6428b52e2837002eab1dbfeccce7e09e60a5bc3b`.
It pins the selected step-1,792 checkpoint hash above and the query-cache
receipt SHA-256 `7f5ffd7362e84770b864a189d50bc29f43be3738f54a07f6dbdc21f1ef7d909e`.
The execution path merges the separately stored new-query bank into the
corpus bank map, matching the fit runner. Existing feature arrays, checkpoints,
and corpus files were read only.
