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
- Output directory: `C:\CodexArtifacts\pretrained-reuse-r2\pch-recording-temporal-attempt-01`
- Status at protocol freeze: `ruff check .` passed and the full CPU suite passed
  2,326 tests (11 skipped). The attempt-01 real-source preflight passed, but its
  guarded child was interrupted before any optimizer update after a CPU smoke
  exposed a checkpoint-key prefix bug. Its directory is preserved; it has no
  candidate checkpoint, feature cache, or training history. A regression test
  and corrected CPU smoke now pass. The corrected code will use attempt-02.

