# Fresh-speaker M1 metadata preflight — 2026-11-10

## Result

The metadata-only split gate passed on attempt 06. The run may proceed to
media materialization only from these frozen plans. Attempts 01–05 were
preserved externally and did not access audio, load a model, use the GPU, or
run inference/training. Attempt 03 exposed an implementation mismatch between
the five-rows-per-speaker cap and a mistaken distinct-speaker-per-class check;
attempt 04 exposed class-order starvation under a sequential sampler; attempt
05 showed that taking the first 200 hash-ranked groups did not cover 100
distinct speakers for `left` (93 were available). The metadata-only protocol
and selector were corrected before any model or media results were observed.

## Frozen source and execution

- Dataset: `google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da`,
  v0.02, official TRAIN shards only.
- Source label catalog verified as the pinned 36-class order; the ten target
  labels are `yes,no,up,down,left,right,on,off,stop,go`.
- Columns read: `label`, `speaker_id`, and `file`; the audio column was not
  requested. No validation/test shard or sealed audit data was read.
- Preflight command:

  ```powershell
  node scripts/preflight_speech_commands_fresh_m1.mjs `
    --historical-dir C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-sample-seed17-128-64-retry2 `
    --observed-manifest C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-fresh-speaker-eval-20261110-attempt-02-corrected\cohort-manifest.jsonl `
    --output-dir C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-fresh-m1-preflight-20261110-attempt-06 `
    --hyparquet-entry C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-fresh-speaker-eval-20261110-attempt-05-runtime\node_modules\hyparquet\src\node.js
  ```

- Metadata transport: 638 range requests, 2,477,844,526 response bytes, six
  shard ETags recorded in `preflight-report.json`.
- Local C: free space after preflight: 5.21 GiB. No media has been written.

## Split evidence

| Measure | Result |
|---|---:|
| Eligible source rows after exclusions | 4,396 |
| Eligible speaker groups | 617 |
| Train rows / speaker groups | 1,500 / 403 |
| Maximum selected train rows per speaker | 5 |
| Fit rows / speaker groups | 1,210 / 322 |
| Calibration rows / speaker groups | 290 / 81 |
| Development rows / speaker groups | 1,000 / 200 |
| Rows per target class, train | 150 each |
| Rows per target class, development | 100 each |
| Historical or previously observed speaker overlap | 0 |
| Historical or previously observed source-row overlap | 0 |
| Historical or previously observed source-path overlap | 0 |
| Train/development speaker overlap | 0 |
| Train/development source-row overlap | 0 |
| Train/development source-path overlap | 0 |

The initial selector's first-200-by-hash development set had only 93 distinct
`left` speakers. The frozen quotas were retained; the corrected selector uses
seeded greedy class coverage for the 200 development groups, then fills each
class to 100 distinct speakers. Training classes are drawn one row per class
per round from per-class SHA-ranked queues, with a five-row total speaker cap.

## Artifact hashes

Artifacts are stored outside Git in:
`C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-fresh-m1-preflight-20261110-attempt-06`.

| File | SHA-256 |
|---|---|
| `train-plan.jsonl` | `1ab0048dd24b32fcbbaeeb2b2528c96832b8d386658140b9526d05d6f656b6e5f` |
| `validation-plan.jsonl` | `2e91f2072090062383bb234a07337c07b13d7382e5a1b22e8552df8eea009da2` |
| `preflight-report.json` | `34dff3dafcd64188a637d336f46f094c2cc334fb29195b4b196bab959dd17ba` |
| `attempt-06.log` | `99463cd478006b6970b5be04a363be906d0e41a08c75d49858c96fa6f31d31b1` |

The report states `audio_column_requested=false`,
`media_downloaded=false`, `model_loaded=false`, `inference_run=false`,
`training_run=false`, `gpu_used=false`, and `sealed_audit_opened=false`.

## Gate

The plans pass the metadata identity checks and frozen quotas. They do not yet
prove WAV availability/content hashes or exact decoded-media integrity. Before
training, materialize only these manifest rows, verify each file and content
hash, reject duplicates across splits, then extract the pinned frozen encoder
features. If media or integrity checks fail, stop without fitting or evaluating
a model.
