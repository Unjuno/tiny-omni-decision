# Reusable native speech decisions and JSON replay

## Implemented path

`NativeSpeechRuntime` receives one caller-owned, verified Whisper model plus its
fixed English tokenizer/processor. It validates mono16kHz int16 PCM without
silent truncation/resampling, encodes once, and reuses that acoustic encoding for
multiple candidate sets. A union of candidate transcripts is decoded once per
group of requests. Mean transcript/EOS log-probabilities exclude prefix/padding;
each supplied option set receives its own softmax and deterministic custom-ID JSON.

No generative JSON decoder, new model copy, calibration or training is added.
Foreign-runtime encodings, duplicate IDs, invalid audio and truncating transcript
lengths fail closed. At most8 candidate sets and16 distinct union transcripts
are accepted. Caller-managed observation caching is not a persistent cache service.

## Real-input parity

First8 fixed existing LibriSpeech assets, both random and hard distractor sets:
16 distinct decisions, plus reversed-order repeats. No new evaluation examples
or sealed audit were opened. Same37,760,640-parameter Whisper-tiny checkpoint.

| Profile | Probability max difference | Raw score max difference | Answers changed | Verdict |
|---|---:|---:|---:|---|
| CPU | 6.91484269044e-06 | 0.000180387248595 | 0 | FAIL_RAW_SCORE_PARITY |
| CUDA | 1.17717613546e-07 | 1.63128501551e-06 | 0 | PASS_NATIVE_SPEECH_JSON |

The reference was computed on CUDA. Both profiles preserve all selected IDs and
pass the fixed1e-4 probability gate. CPU fails the separately fixed1e-4 raw-score
gate (maximum0.00018039); this threshold was not relaxed. Matching the reference
CUDA backend passes both gates. This supports a backend-numerics explanation,
not a claim of bit-exact CPU/GPU equivalence. No new accuracy claim follows.

On CUDA, forward hooks verify8 acoustic-encoder calls and16 decoder calls:
8 union batches for normal options plus8 for reversed options. These service32
candidate-set evaluations including repeats. Reordering preserves selected IDs.
CPU call count16 was initially inferred from the loop; CUDA verifies it directly.
Parameter version counters and on-disk model hashes remain unchanged.

## Failures retained

- The first invocation had a runner syntax error, corrected before any model execution.
- The default Python environment lacked SoundFile. The already existing isolated
  native-speech dependency directory was reused; nothing was installed/upgraded.
- Attempt01 stopped at asset5 on the raw-score tolerance. Its config and failure
  record remain. Attempt02 aligns union candidate batching and records the full
  CPU failure; its exact executed source is archived. Attempt03 is the fixed
  matching-backend CUDA replay. Previous artifacts were not overwritten.

## Runtime scope and remaining gates

This reusable path recognizes which supplied transcript fits English speech.
It does not establish general audio-event understanding, spoken-condition
reasoning, or joint audio/video fusion. The caller must verify the model/processor
revisions and must not mutate them between encode and score. Sensor and combined
Omni routing remain unfinished;95–98% product quality is not demonstrated.

CPU measured replay loop5.535s; CUDA0.873s for these8 assets, excluding model load.
This single small sample is not a p95, cold-start or hardware performance study.
Peak memory was not measured in these replays. GPU use was inference only, on
local RTX3080Laptop; no training, cloud cost or pre-existing process interruption.

## Reproduction

Use the existing dependency overlay:
```powershell
$env:PYTHONPATH='C:/CodexArtifacts/pretrained-reuse-r2/native-speech-deps-01;C:/CodexWorktrees/pretrained-reuse-omni/src'
python -m scripts.replay_native_speech_json --device cuda --output C:/CodexArtifacts/pretrained-reuse-r2/native-speech-json-replay-03
```

The saved directory exists; reproduction requires a fresh output path.
CUDA run source `b8b6696`. Actual model/tokenizer/processor hashes, IDs, tolerances
and predicted JSON are retained under `native-speech-json-replay-01/02/03`.
Adjacent JSON contains configs and results, including the CPU failed gate.

## Verification

Ruff passed; 590 CPU tests passed, 1 skipped. Model/dataset/catalog validation
passed. Target-commit GitHub Actions is checked after push.
