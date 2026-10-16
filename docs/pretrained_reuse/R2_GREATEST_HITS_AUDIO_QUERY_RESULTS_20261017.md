# Frozen audio/query execution before the AV gate comparison

## Executed

Implemented in managed worktree
`C:/Users/junny/.codex/worktrees/r2-av-audio-cache/tiny-omni-decision`, branch
`codex/r2-av-audio-cache`, source
`61530adc76935e4b5dbfbabf582ef1ec03cd1405`. The running video checkout remains
unchanged. New pure helpers have eight RED-to-GREEN tests for label-independent
all-option queries, explicit nominal audio time coordinates, exact-once bank
writes and invalid writes failing before mutation.

The original 23,707 TRAIN / 8,875 validation questions, 8,102 / 3,049 distinct
events, source media and encoder weights are preserved. Queries include all
17 material, two action and five reaction options. Correct labels never select
which query or event representation is encoded. No optimizer, fitted readout,
prediction-quality evaluation, calibration or sealed audit was used.

Actual smoke: original 96 kHz mono media -> complete nearest-sample 0.5-second
crop -> independent direct 16/32 kHz resampling -> frozen Whisper/EfficientAT.
All 24 original question/candidate pairs pass the frozen DeBERTa pair encoder.
FP32/eager, seed17, deterministic algorithms, TF32 off, all modules eval/frozen.
The same source is rerun in a fresh process with a fresh output directory.

| Measurement | Smoke | Fresh replay |
|---|---:|---:|
| Events / parents / represented questions | 21 / 21 / 63 | 21 / 21 / 63 |
| Query vectors | 24 | 24 |
| Wall seconds | 31.3008 | 29.9776 |
| Audio forward seconds | 2.5045 | 2.1245 |
| Native audio decode seconds | 2.0462 | 2.1642 |
| Query forward seconds | 2.5413 | 1.7885 |
| Peak CUDA allocated bytes | 485,888,000 | 485,888,000 |
| Peak process working set bytes | 2,170,822,656 | 2,175,139,840 |

The 45 serialized tensors and the event/query plan match exactly. This includes
42 per-parent audio banks, two shared position arrays and one query bank. Both
runs preserve all model state hashes. All 21 decoded native crops match the
earlier TRAIN source review. Exactly two events coincide with the earlier
three-event audio probe: direct resamples and encoder features match exactly.
The third probe has another event in the same parent and is not counted as an
overlap. All 21 smoke event identities and nearest-sample clocks independently
match the previously saved video smoke.

Loaded parameters: Whisper 8,208,384; EfficientAT 1,961,239; DeBERTa 70,830,722;
mel frontend zero. This feature work does not change the existing 196,254,874
Attention baseline or the 197,097,949 allocated exchange/gate comparison model.
No inference model is promoted. Gates remain untrained on joint AV tasks.

## Runtime, commands and evidence

Existing local RTX3080 Laptop16GB, i7-12700H, Python3.11.9, PyTorch2.6.0+cu124,
Transformers5.6.2, NumPy2.4.6 and PyAV18.1.0. Exact package versions, model
revisions, source/processor/weight hashes, commands and timestamps are in each
receipt. No environment upgrade, cloud compute or cloud cost. Video CPU caching
was concurrent; these timings are not uncontended performance benchmarks.
The CUDA number is allocator peak, not total device memory used by the process.

Run from the managed checkout with existing system Python:

```powershell
$env:PYTHONPATH='C:/CodexArtifacts/pretrained-reuse-r1/python-deps;C:/CodexArtifacts/pretrained-reuse-r2/native-speech-deps-01;C:/Users/junny/.codex/worktrees/r2-av-audio-cache/tiny-omni-decision/src;C:/Users/junny/.codex/worktrees/r2-av-audio-cache/tiny-omni-decision'
$env:CUDA_VISIBLE_DEVICES='0'
$env:OMP_NUM_THREADS='4'
& 'C:/Users/junny/AppData/Local/Programs/Python/Python311/python.exe' -m scripts.cache_greatest_hits_audio_query --phase smoke --output C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-audio-query-smoke-01
& 'C:/Users/junny/AppData/Local/Programs/Python/Python311/python.exe' -m scripts.cache_greatest_hits_audio_query --phase smoke --output C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-audio-query-smoke-replay-01 --reference C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-audio-query-smoke-01 --reference-sha256 95686ac92f430a980d453f3602b8bdd7404c49914393eeaab404fe6c4b68bfea
```

These output paths already exist; the runner rejects reuse. Do not rerun these
commands into them. Both smoke processes completed with exit code0.

Under `C:/CodexArtifacts/pretrained-reuse-r2/`:

| Receipt | SHA-256 |
|---|---|
| `greatest-hits-audio-query-smoke-01/result.json` | `95686ac92f430a980d453f3602b8bdd7404c49914393eeaab404fe6c4b68bfea` |
| `greatest-hits-audio-query-smoke-replay-01/result.json` | `5bbbdbd64ab9f516a45d7ea32f229b7591fc4d12db5fb85e2e8a0fd5d89e03e0` |

Independent audit in `greatest-hits-audio-query-verification-01/audit.py`:
SHA `3a8b40aaddd6c1cfcf07c055c32648617f15ed980fdf88bb4959390bac3f1914`;
status `PASS_AUDIO_QUERY_SMOKE_REPLAY_AUDIT`. All8,714 earlier immutable files
remain unchanged and94 new files (two receipts plus46 outputs each) are verified.
`artifact-hashes.json` binds those files. It excludes both active full caches.
The small repository manifest
`manifests/pretrained_reuse/greatest-hits-audio-query-20261017.json` binds the audit.

## Running, not completed

After both passing smoke receipts, launched exactly:

```powershell
& 'C:/Users/junny/AppData/Local/Programs/Python/Python311/python.exe' -m scripts.cache_greatest_hits_audio_query --phase full --output C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-audio-query-cache-01 --smoke C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-audio-query-smoke-01 --smoke-sha256 95686ac92f430a980d453f3602b8bdd7404c49914393eeaab404fe6c4b68bfea --smoke-replay C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-audio-query-smoke-replay-01 --smoke-replay-sha256 5bbbdbd64ab9f516a45d7ea32f229b7591fc4d12db5fb85e2e8a0fd5d89e03e0
```

Same environment above. Source61530ad; actual PID12824; exec session26274;
started2026-10-16T18:47:45.883981+00:00. Log
`greatest-hits-audio-query-verification-01/full.log`. At an interim observation,
594/11,151 events,28/409 parents and1,724/32,582 questions were saved. This is
progress evidence only. Require the final receipt, complete counts, source and
frozen-state checks before calling the cache complete. A partial cache is not
eligible for training. No resume is claimed.

The video job remains PID17848/session92865, output
`greatest-hits-video-cache-01`; no stop/restart/code replacement. Both jobs have
fixed resource guards. Do not modify either executing checkout's Python/config
or previously saved artifacts. Future docs-only publication does not change the
Python/config pins checked by the running jobs.

## Verification and remaining work

Ruff passes; CPU suite2,275passed/1skipped; all25 existing manifest commands pass.
Logs and command receipts are saved in the verification directory. These checks
include the8 new tests. GPU smoke/replay ran separately from tests. Publication
CI is recorded separately in `greatest-hits-audio-query-verification-01/ci.json`
for the pushed publication commit; the earlier branch's CI is not this proof.

Next finish and audit both full caches; validate explicit physical event positions
and content-keyed candidate transport in the Decision input path. Then freeze
equal budgets, initial tensors, sampling and validation selection for the four
Attention/exchange/gate ablations. Compare valid paired input interventions and
correctness, not gate magnitudes alone. Retain missing-input behavior and the
original baseline.

This dataset supports scoped native material/action/reaction development. Silent
video annotation does not certify both-required answerability or physical AV
sync. Nominal speech token positions are frontend coordinates, not local semantic
receptive fields. No result here demonstrates fusion gain, robustness to conflict,
cross-task generalization, or the four-modality95–98% product target.
