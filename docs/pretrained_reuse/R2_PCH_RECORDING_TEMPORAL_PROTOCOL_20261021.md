# PCH recording-level temporal order diagnostic — 2026-10-21

## Question and scope

Does preserving the order of adjacent 2.5-second accelerometer windows improve a
small text-free activity decision readout beyond the already measured
window-level LSTM? This is a sensor-only architecture diagnostic motivated by
the project's interest in temporal state. It is not a multimodal or product
quality result.

The frozen reference is the best checkpoint from
`pch-lstm-sensor-attempt-04`: the 64-hidden-unit window LSTM selected at epoch
31, with 18,834 parameters. Its window LSTM representation weights and
fit-only normalization will be loaded and frozen. A new 32-hidden-unit
recording-level LSTM will aggregate the ordered embeddings; only this small
readout and its 18-class output layer are trained.

## Frozen comparison

Two seed-17 arms use the same frozen encoder, participant split, labels, fit and
validation records, classifier initialization, record-batch order, optimizer,
learning rate, stopping rule, and checkpoint selector:

1. `chronological`: each recording's window embeddings stay in source-record
   order after the upstream timestamp segmentation.
2. `shuffled_control`: each recording is permuted once by a deterministic
   SHA-256-derived seed (`17:sample_id`) and uses that same permutation during
   fitting and evaluation.

Both arms use AdamW (`lr=0.001`, `weight_decay=0.01`), batches of 16 recordings,
up to 100 epochs, early-stopping patience 10, `min_delta=1e-5`, and minimum
validation recording NLL selection. Every epoch covers every fit recording once.
No seed search, added data, encoder update, validation change, or hyperparameter
search is planned. The existing window-LSTM validation predictions are the
matched reference. Reverse-order and shuffled-order evaluations are diagnostic
perturbations of each selected checkpoint.

Report Accuracy, present-class macro Accuracy, NLL, Brier, ECE-15, mean
confidence, and class support/Accuracy. Compare chronological against fixed
shuffle and the window-LSTM reference using paired participant-cluster bootstrap
(5,000 resamples, seed 17). Validation has already selected the original
window-LSTM checkpoint and is development data; these paired intervals do not
turn it into an independent test.

## Data, isolation, and preservation

Use only the existing PCH Data 2017 archive and the existing attempt-04 window
LSTM baseline. The deterministic split remains 276 fit recordings / 92
participants and 63 validation recordings / 21 participants. The 28-participant
holdout is not opened; its labels and signal payloads remain unused. No global
sealed audit is loaded. Existing checkpoint, training history, and run artifacts
are read-only. All new run output is external under a new attempt directory,
and no model weights or media are committed.

The input sequence represents file order across timestamp-segmented windows.
No explicit elapsed-time or gap embedding is added, so this does not test
long-range timing across irregular gaps. Validation improvement, if any, remains
exploratory until a separately frozen evaluation on untouched participants.


## Preserved attempt-01 correction

The attempt-01 source preflight passed and pins the original train/validation
IDs. Its first HotUGPU invocation rejected a bare `python` token before launch
because the wrapper requires a resolvable executable path. The corrected
absolute-path invocation waited for cooperative GPU yield and reserved 5,120
MiB. Before any optimizer update, an independent CPU smoke found that the
window-encoder loader stripped the checkpoint's required `lstm.` state-dict
prefix. I interrupted only this task's own guard/child. The wrapper released its
reservation; Orbitune was not signaled or stopped. Attempt-01 remains preserved
with preflight/config/command only and must not be resumed because its pinned
script hash is stale.

A new regression test exercises the checkpoint's `lstm.*` keys, and the corrected
loader now passes a CPU-only smoke using the real PCH train/validation archive
and the frozen attempt-04 checkpoint. That smoke embedded eight train/validation
records, trained a two-epoch temporary readout, and exactly reproduced its
selected checkpoint predictions. The 28-person holdout remained unopened.

## Implementation and execution record

- Code: `scripts/pch_recording_temporal_probe.py`
- CPU regression tests: `tests/test_pch_recording_temporal_probe.py`
- Preflight: hash-checks the exact PCH archive, baseline checkpoint, baseline
  preflight/result/predictions, split, window order and validation IDs.
- Training uses the existing reviewed cooperative `run_with_hotgpu_guard.py`
  wrapper. It waits for external GPU processes to yield and reserves local VRAM
  before starting the child; it does not stop or signal Orbitune.
- Attempt-01 exposed and preserved the checkpoint-key prefix bug described
  above. Attempt-02 preserved a preflight with the wrong experiment ID and a
  placeholder interpreter command; it was not run. Attempt-03 verified the
  corrected metadata but the waiting guard was cancelled during the handoff;
  it contains no run results. No attempt directory was overwritten.
- The corrected run used the absolute Python 3.11 executable and the pinned
  `run_with_hotgpu_guard.py`. It started only after Orbitune voluntarily yielded
  and a 5,120 MiB reservation was acquired. It completed on the local RTX 3080
  Laptop GPU without cloud cost. No Orbitune process was signalled or stopped.
- Code source commit: `3409550269a50579b432c0e1bf229b8a933c3e4b`;
  script SHA-256: `53d4d4b97f96686070507e6664d1530995371de4283d67a060d636e94ac51202`.
  The effective config binds the experiment ID to the external attempt directory.
  CI passed on this source commit; local targeted tests passed (21 tests) and
  Ruff passed.
- Attempt-04 preflight SHA-256:
  `82427c3ce2534ea4c68d119b5a076cb7ecb14b97c22e8fdd1c7c5ef8c934a6c7`;
  effective-config SHA-256:
  `d05ac600041f65fb433452a604be73f7a48de5b732343568dda66dcca1a50094`.
  Window-order hash remained
  `3774642255f83443a994f9a23de1c0c9a4bbca4085bbd9fde06f34401e4a6980` and
  validation ID-order hash remained
  `32ba9659a2b6cec52812ddb4ac0bd60fab073f39ba92901d04c056102d12d3f9`.
  The 28-participant holdout and sealed audit were not opened.

## Attempt-04 measured result

The run completed both arms with best-checkpoint reload equivalence verified.
Each epoch visited all 276 fit recordings (92 participants; 13,100 windows).
The two arms consumed 17,940 recording presentations over 1,170 optimizer
updates. Validation used the same 63 recordings (21 participants; 3,041
windows). Total measured wall time was 13.17 seconds, including 0.46 seconds
for feature extraction. Peak training allocation was 77,602,304 bytes allocated
and 211,812,352 bytes reserved; feature extraction peaked at 195,506,688 bytes
allocated and 209,715,200 bytes reserved. Environment: Python 3.11.9,
PyTorch 2.6.0+cu124, CUDA runtime 12.4, NVIDIA GeForce RTX 3080 Laptop GPU.
PyTorch emitted its known Windows warning that `expandable_segments` is not
supported; the guarded run still completed and produced finite metrics.

| Selected checkpoint on the fixed development validation | Accuracy | Present-class macro Accuracy | NLL | Brier | ECE-15 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen window-LSTM baseline | 65.08% | 62.22% | 1.0117 | 0.4153 | 0.1188 |
| Recording LSTM, chronological | 63.49% | 62.22% | 1.2019 | 0.4641 | 0.1582 |
| Recording LSTM, fixed shuffled control | 65.08% | 63.61% | 1.1668 | 0.4574 | 0.1207 |

The chronological arm selected epoch 23 and stopped at epoch 33; the shuffled
arm selected epoch 22 and stopped at epoch 32. Both best checkpoints reloaded
to exactly reproduced predictions. In the chronological arm, train CE fell
from 0.5125 at the selected epoch to 0.3298 at stopping while validation NLL
rose from 1.2019 to 1.2511. In the shuffled arm, train CE fell from 0.5230 to
0.3979 while validation NLL rose from 1.1668 to 1.2454. This is evidence of
overfitting within these runs, not a measured generalization gap on a newly
independent set.

Participant-cluster bootstrap used 5,000 paired resamples of 21 validation
participants (seed 17). Chronological minus fixed-shuffled Accuracy was
-1.59 percentage points with 95% percentile interval [-4.76, 0.00] pp; the NLL
difference was +0.0350 with interval [-0.1148, +0.2460]. Chronological minus
the frozen window-LSTM Accuracy was also -1.59 pp with interval [-9.52, +6.35]
pp; NLL difference was +0.1902 with interval [-0.0053, +0.3761]. The intervals
are wide and overlap zero. Order perturbations changed selected predictions
in 7.9–11.1% of recordings, but the fixed shuffled control was not worse than
chronological training. This establishes some order sensitivity, not useful or
robust temporal reasoning.

**Interpretation: the proposed recording-level temporal readout did not show a
clear benefit on this development split.** It did not improve over the frozen
window-LSTM baseline on Accuracy and worsened NLL/Brier/ECE. The run does not
show that recurrent architectures cannot help; the sample is small, the
validation cohort has already influenced baseline checkpoint selection, and
only one seed was used. No holdout, multimodal, or product-quality claim follows.

The complete output is external at
`C:\CodexArtifacts\pretrained-reuse-r2\pch-recording-temporal-attempt-04`.
Key SHA-256 values: `result.json`
`94b4be48cd28cd300016b0f2decdbc16bf8157a5e491e34f83d6b98914bff112`,
`validation-predictions.jsonl`
`ccd4a07e5ac51932972a0c81fd133432e3a191f85e38cb44ae806264e66fa4eb`,
chronological best checkpoint
`ecca5d931a391f489732a0ccc05d927f602e06a20375b7ce5e36a1cf68867b6e`, and
shuffled-control best checkpoint
`8cea3566412b4b1dc910fa6195f238dabb6246cbaa16d47f863b9816344d6b87`.
Weights, media, and feature cache remain outside Git.

