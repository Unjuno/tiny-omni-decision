# Full audio cache: failed final count, independently verified saved outputs

## Actual failure and cause

Producer source61530ad saved all11,151 events,409 parents and32,582 represented
questions, then exited1. Its status remains **INCOMPLETE**, unchanged. The final
compound check expected three matches against the earlier three-event audio probe;
the actual count is two. Every other term of that check (coverage,21 review/21
smoke references and before/after encoder state) holds.

The third reference is parent`2015-03-30-00-26-27`, source line1, time1.778125.
Its original annotation is `None None None`. No frozen native-field question was
created for it. The later source line3 of that parent is in the21-review cohort.
The reference count therefore must be the selected-event/reference intersection,
not the size of the earlier probe. Adding an unlabelled event to the fixed corpus
would change the experiment and was not done.

Source`933e5dc84cbf125b3a2965f4250fdd0de5ad3ef5` fixes the future producer to compute
and save the expected overlap. Four new failing-first regression cases cover
excluded/included references and duplicate selected events.12 audio helper tests
pass; full CPU suite2,297passed/1skipped, Ruff and25 manifest commands pass. Two
line-length lint findings in the new verifier were fixed before its execution.

## Independent full verification

The new read-only CPU verifier accepts only the exact pinned failed producer and
error. It does not edit that result, reload an optimizer, retrain, or run the
encoders again. A fresh process redecode/resample pass and full bank inspection
completed with exit0 and status **PASS_FULL_AUDIO_CACHE_REVALIDATION**.

| Check | Verified |
|---|---:|
| Parents / source media | 409 |
| Frozen events | 11,151 |
| Represented original questions | 32,582 |
| Fresh96k crop / direct16k+32k input-digest matches | 11,151 |
| Stored speech/event feature-digest matches | 22,302 |
| Known smoke feature slices exactly matching | 42 (21 events ×2) |
| Original producer files hashed | 823 |
| Older immutable files unchanged | 8,824 |

The complete plan equals the frozen corpus; source/parent/split/line/index/order,
all nearest-sample clocks, finite FP32 bank shapes, query/position banks, original
code/model/input pins and frozen model state all pass. The omitted reference is
recorded with its original annotation hash. Audio/model/data/label files are not
modified. This verifies all stored input replays and captured feature digests; it
is not a fresh full encoder recomputation or a quality evaluation.

| Resource | Producer (failed final assertion) | Independent verifier |
|---|---:|---:|
| Wall seconds | 1,576.3698 | 241.6151 |
| Audio forward seconds | 1,010.1278 | No encoder forward |
| Native decode seconds | 47.4555 | Included in wall |
| Peak CUDA allocated bytes | 485,888,000 | CUDA hidden |
| Peak process working set bytes | 1,971,634,176 | 331,710,464 |

Existing RTX3080 Laptop GPU / CPU environments were reused. Producer uses frozen
FP32 Whisper/EfficientAT/DeBERTa; verifier uses existing Python3.11.9/PyAV18.1.0/
NumPy2.4.6 with one decoder thread. Video CPU extraction was concurrent. No cloud
cost, model learning, validation-quality tuning, calibration or sealed audit.
Audio PID12824 terminated and was later reused by an unrelated node process;
monitor handles/command/start time, not the old number alone. No process was killed.

## Artifacts and command

All directories below are under `C:/CodexArtifacts/pretrained-reuse-r2/`.

| Receipt | SHA-256 |
|---|---|
| `greatest-hits-audio-query-cache-01/result.json` (still INCOMPLETE) | `74a12e2d2779751eed6867b3fd0186e6ad14d9ea44a527a2d9b0ceee41a1c10d` |
| `greatest-hits-audio-cache-revalidation-01/result.json` | `4b273e75298e640b79cbe5de0e234f1d1b6f54e45965af692cba86471b95f4cb` |

Run from the isolated transport checkout at933e5dc:

```powershell
$env:PYTHONPATH='C:/CodexArtifacts/pretrained-reuse-r1/python-deps;C:/CodexArtifacts/pretrained-reuse-r2/native-speech-deps-01;C:/Users/junny/.codex/worktrees/r2-av-transport/tiny-omni-decision/src;C:/Users/junny/.codex/worktrees/r2-av-transport/tiny-omni-decision'
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'
& 'C:/Users/junny/AppData/Local/Programs/Python/Python311/python.exe' -m scripts.revalidate_greatest_hits_audio_cache --output C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-audio-cache-revalidation-01
```

The output exists and cannot be reused. Exact execution/source/pins/timestamps
are in its receipt. Original failure log remains at
`greatest-hits-audio-query-verification-01/full.log`; new verification log is
`greatest-hits-audio-full-verification-01/revalidation.log`.

Publication audit `greatest-hits-audio-full-verification-01/audit.py` SHA
`b3cc2419d1c04bb33681dd0daad5b44252b89cf8c81ae27cfcee10ea6028c753` verifies825 new
files (823 producer files plus the new result/ledger), and all8,824 earlier files.
Its result status is PASS_FULL_AUDIO_REVALIDATION_PUBLICATION_AUDIT. The repository
manifest `manifests/pretrained_reuse/greatest-hits-audio-revalidation-20261017.json`
binds the receipts/ledger. Exact publication CI is recorded separately in
`greatest-hits-audio-full-verification-01/ci.json`.

## Remaining gates

Raw INCOMPLETE receipts continue to fail the existing transport gate. Future
training readiness must explicitly pin/validate the new independent revalidation
and keep its producer-failure provenance. Do not relabel the original run as PASS.
Video PID17848/session92865 is still running its original source/output; at an
interim check2,838/11,151 events were saved. Complete video integrity and joint
binding are still required before learning. No partially filled video bank is
admitted. No gates were trained and no quality/retention/product target was met
by this cache repair.
