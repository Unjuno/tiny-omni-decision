# GRID / Lombard cross-source JSON interface smoke

The [fixed smoke](R2_GRID_CROSS_SOURCE_INTERFACE_PROTOCOL_20261028.md) used the
first existing Lombard **train** audio clip and the first four fixed GRID `s1`
videos. These are deliberately unmatched sources: no correct target or
Accuracy was defined, and this does not establish audiovisual quality.

The real 16 kHz audio contained 35,200 samples. A locally pinned, frozen
8,208,384-parameter Whisper encoder and existing audio head produced finite
51-slot logits on CPU. Four video logits came from the separately verified
GRID native VideoPrism and frozen video-head runs; those models were not rerun
here. The existing `NativeAVCorrespondence` accepted the real audio and video
bytes and returned four opaque-ID probabilities summing to
`0.9999999999999998`. The runtime recorded one audio backend call, four video
lookup calls and one repeated-audio cache hit. The video calls in this run are
hash-checked cached-logit lookups, not new encoder forwards. There was no
training or GPU use. The reported 0.816 s covers the script's body after its
initial source-hash checks and is not end-to-end cold latency.

Output:
`C:/CodexArtifacts/pretrained-reuse-r2/grid-cross-source-json-20261028-attempt-01/`;
`summary.json` SHA-256
`f33039d4e58a612051e7d359619aaf031319693311bd4daee083b4c194de052f`.
All source hashes are in the receipt. A JSON validation test and Ruff passed.
Matched GRID audio/video extraction, timing validation, actual candidate scoring,
speaker independence and any quality gate remain open.
