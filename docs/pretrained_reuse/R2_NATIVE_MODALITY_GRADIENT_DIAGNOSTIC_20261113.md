# Native modality-gradient diagnostic — 2026-11-13

## Result

The fixed 256-update TRAIN window around the already selected native baseline
checkpoint completed as `PASS_NATIVE_MODALITY_GRADIENT_DIAGNOSTIC`. The
four-modality gradients on the shared readout are weakly aligned and frequently
conflicting; the simulated equal-modality batch mean also reaches the baseline
global clip threshold almost every update. This makes optimization interference
a plausible contributor to the historical native-readout result. It does not
show that interference is the only cause, and it is not a Path A quality or
product result.

| Measure | Result |
|---|---:|
| Updates / examples | 256 / 2,048 |
| Unique question IDs / repeated occurrences | 1,762 / 286 |
| Unique non-text assets / modality-group identities | 977 / 1,121 |
| Examples per modality | 512 each |
| Mean pairwise all-parameter cosine | 0.019–0.033 |
| Pairwise all-parameter cosine below zero | 39.1–45.3% of updates |
| Mean shared-parameter cosine | 0.021–0.036 |
| Pairwise shared cosine below zero | 39.1–45.3% of updates |
| Existing global clip norm | 1.0 |
| Updates reaching global clipping | 254/256 (99.22%) |
| Global clip scale, median / mean | 0.551 / 0.560 |
| Shared-gradient cancellation ratio, median / mean | 0.580 / 0.584 |

The 2,048 occurrences came from sampler updates 7,937–8,192, immediately
preceding the validation-selected step-8,192 checkpoint. Source mix was
CLEVR-4 512, CLEVRER 512, STARSS 240, Lombard 272, text original 49, text
curriculum 130, condition relevance 168, and programmatic text decisions 165.
There were 286 repeated sample occurrences in this window; it is not a unique
example-only estimate. The fixed sample-order digest is
`bfc8f17ffc05f7a96c1b181b68b408b066dfdaad96b985548558b4be179bc19f`.
The immutable raw outputs are stored at
`C:\CodexArtifacts\pretrained-reuse-r2\native-modality-gradient-diagnostic-v1-attempt-07`.

## Scope and integrity

- Baseline: commit `8756a09f582554c0321ed5ece71846d6b19469cd`, 32,768-update
  native frozen-encoder run; selected checkpoint step 8,192.
- Checkpoint SHA-256:
  `d7b666cd7ea60dcf236bd06e53b0107e0d5f74193824c1f4b44bbe4d5cdc1b56`.
- Selected model: 889,345 readout parameters; FP32.
- Diagnostic source commit: `07e98ed05f6ceba213dd6d39b706eedef8e05ff3`.
- Config SHA-256:
  `4736f402261a2bcb07caf7e566af98c2d81daa0e76f42f285077109d8b0f6d1e`.
- Runtime: Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4,
  RTX 3080 Laptop GPU. Wall time 29.79 s; peak allocation 204,413,952 bytes.
- Run time: `2026-11-13T09:34:36Z`–`2026-11-13T09:35:05Z`.
- Model state SHA-256 before and after the measurement is identical:
  `77b3b1c3a38ab5cd7cad58b1a4b9d60185dbae4b25b7b626ab614088ce3a99b2`.
- `optimizer_updates=0`; no validation, calibration, or sealed-audit rows,
  labels, features, or predictions were loaded. Cloud cost was $0.
- Attempt-07 `result.json` SHA-256:
  `0b9549125b446197932c733e611f1ce6637c4fd6ec2e1fa44e02d588cb9f55bd`.
- Attempt-07 `updates.json` SHA-256:
  `bf51ca6d5e63935d1a684d6274aa6e333db1d5b5a387460740f1d6bfb5d33bf1`.
- Attempt-07 `window-accounting.json` SHA-256:
  `8149c943810f9f0f901f33bc480b527016d689f9b4194711bd78b5a29ab03077`.
- Attempt-07 `sample-ids.json` SHA-256:
  `a69a000f17cdbe58388a1604dec80b0844b144c381ecb1cddc563e2804a8f727`.

The source checkpoint had already been selected by the historical validation
selector. This diagnostic observes only its immediately preceding TRAIN
window; it is not an independent estimate of generalization or an unbiased
sample of the full learning trajectory. The pairwise gradient statistics are
descriptive and do not establish that gradient conflict caused any specific
validation error.

## Attempts and code correction

All attempts used distinct immutable directories under
`C:\CodexArtifacts\pretrained-reuse-r2\native-modality-gradient-diagnostic-v1-attempt-XX`.
Attempts 01–06 failed before any optimizer update; all failure receipts are
preserved. They exposed, in order: an implicit CUDA device index, a mistyped
pinned query-cache digest, a one-sample GPU preflight transient, a missing
language query bank, missing deterministic cuBLAS workspace configuration,
and an overstrict shape check that incorrectly required the all/shared/private
gradient vectors to have equal lengths. The last issue is now covered by a
regression test: each comparison scope validates its own cross-modality shape,
while all scopes must remain on one device. Attempt-07 passed after this fix.

Focused verification on the final code: 10 tests passed, Ruff passed, CLI help
passed, and `git diff --check` passed before commit. The result and code fix are
on `codex/r2-native-gradient-diagnostic-20261113` at
`07e98ed05f6ceba213dd6d39b706eedef8e05ff3`.

## Decision

This result is specific to the earlier jointly trained native readout. It does
not change the Roadmap's frozen modular Path A results, justify another
full-backbone or LSTM training run, or establish that a shared-lens / weight
merge experiment will help. The separately measured Path A candidates remain
the relevant R2 quality evidence: MiniWoB DINO is a narrow synthetic screen
task, while VideoMAE spatial pooling improves a reused CLEVR-4 development
selector but remains weak on count. Neither result establishes 95–98% across
modalities or true cross-modal necessity. Keep R2-M open until a rights-cleared,
independently grouped task demonstrates that both synchronized inputs are
needed and improves over matched unimodal controls. Do not use sealed audit
data or repeat exposed-selector tuning.
