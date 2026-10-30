# Compact DeBERTa NLI real-load audit

Pinned publisher checkpoint `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`
revision262ae02f29173eec1c250f90804dc7edc677dcff loaded locally onCPU, without
remote code, missing/unexpected/mismatched keys or dependency upgrades.
Safetensor hash56b89067a4e60362f498f39b7067020911618fad37d64fa23a5f959a04fd4080
matches the Hub LFS digest. Actual loaded and header counts both70,830,722.

Embedding49,191,168 + encoder21,490,944 + pooler/classifier148,610.
Count all embeddings: this is not a22M total model. The card's vocabulary-count
wording is inconsistent with the actual tensors. Checkpoint141,685,964 bytes is
FP16; loadedFP32 parameter bytes283,322,888. Precision changes storage, not counts.
Two synthetic sentence-pair forwards are finite and repeat exactly;1.795s CPU
load/forward scope,2threads. This verifies execution only, not Decision quality.
The first attempt failed JSON serialization of empty loading-info sets after
successful checks; failure retained and serialization fixed with a regression test.

## Full model budget implications

With VideoPrism114,365,184 and Whisper37,760,640:222,956,546 before readouts,
already over200M. This cannot simply replace the current22.7M text tower.
With existing TinyCLIP83,101,185 and Whisper:191,692,547 before readouts.
That alternative leaves8,307,453 parameters, but has no verified temporal path,
combined runtime or sufficient visual quality. Neither sum proves product fit.
Do not count only active layers or exclude embeddings to make the budget pass.

## Provenance / rights / limitations

[Publisher card](https://huggingface.co/MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33/blob/262ae02f29173eec1c250f90804dc7edc677dcff/README.md)
declaresMIT and refers to the larger variant's training-data documentation.
That documentation describes diverse dataset licenses. No underlying datasets
were downloaded, accepted for this project's training, or independently re-audited.
The small card reports supervised-task results and explicitly lacks a separate
zero-shot evaluation; those numbers cannot establish our95%general objective.
Local artifact: `C:/CodexArtifacts/pretrained-reuse-r1/deberta-xsmall-zeroshot-audit-01`.
Runtime/header/count hashes and versions are in JSON. No project training or
product promotion. Next bounded check is the unchanged existing256-question
numeric NLI diagnostic, with native entailment index0 verified before scoring.
