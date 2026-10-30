# Greatest Hits frozen-feature cache and long-fit follow-up — 2026-10-17

This follow-up resolves the earlier live-cache status from the [full-extent
execution record](R2_GREATEST_HITS_FULL_EXTENT_RESULTS_20261017.md). It records
what was already executed and independently rechecked; it does not launch a new
training run or promote an artifact.

## Current-state and process audit

- The roadmap worktree was clean at `2c5232d12c0ec65749be4b96a0231d573ced9f8c`
  and matched `origin/docs/pretrained-reuse-roadmap-20261010` after fetch.
- The separate user's `codex/teacher-quality-clean-dev-v1` worktree contains
  uncommitted files and was left untouched. The other listed worktrees, including
  the QAT and Teacher branches, were not changed.
- GitHub's public PR listing showed PR #15 open/draft; PRs #9–14 are also open
  (PR #13 non-draft). No PR was merged. `gh` CLI authentication is unavailable,
  so authenticated review/check details were not inspected here.
- The historical cache PID 17848 is no longer present. The complete result and
  terminal log are present; no `cache_greatest_hits_video` process was running
  during this audit. An unrelated MIDI/GPU-watcher process was observed and left
  alone.
- The `greatest-hits-video-cache-01` result SHA-256 is
  `9fdb5bfcf75a35c73d3b3634e1affbeb0373074b8cc9240f6567548aba8922ce`.
  Result status is `PASS_COMPLETE_VIDEO_CACHE`, source commit
  `857edcf6f243fb23cd2d34d812036e8b43205dfb`, source VideoPrism revision
  `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`, CPU FP32, zero optimizer updates,
  no sealed audit, and unchanged frozen encoder digest.

## Video bank integrity

The completed run records 11,151 events, 409 parent videos, 32,582 questions,
and all 21 fixed reference events matched. The model used 114,365,184 parameters.
The measured run lasted 13,655.7 seconds (about 3 h 47 min), with peak working
set 4,290,121,728 bytes; cloud cost and GPU training were zero.

A separate read-only audit streamed and rehashed all 1,227 video/position/PTS
files against the run receipt: **70,430,943,552 bytes read, all hashes matched,
67.062 seconds**. The completed result reports `validation_media_loaded=true`,
`sealed_audit_loaded=false`, `quality_evaluated=false`, and
`optimizer_updates=0`. This closes feature-cache execution and integrity only;
it is not a Decision-quality result.

## Audio and fixed-query bank state

The audio/query cache contains both native event-audio paths and fixed query
features for the same 11,151-event / 32,582-question development corpus. The
producer receipt (`greatest-hits-audio-query-cache-01/result.json`, SHA-256
`74a12e2d2779751eed6867b3fd0186e6ad14d9ea44a527a2d9b0ceee41a1c10d`) says
`INCOMPLETE`: it expected three smoke-reference events although the frozen
reference corpus contains two matching source rows. Do not relabel that producer
run as a pass.

The follow-up `PASS_FULL_AUDIO_CACHE_REVALIDATION` receipt
(`greatest-hits-audio-cache-revalidation-01/result.json`, SHA-256
`4b273e75298e640b79cbe5de0e234f1d1b6f54e45965af692cba86471b95f4cb`) independently
verified all 11,151 inputs, 409 parents, 32,582 questions and 22,302 saved audio
feature digests; it reconciled the expected reference count to two. The audit
explicitly records `encoder_replayed=false`. Separately, the 21-event smoke
replay matched 42 saved audio feature payloads, and its audit passed. The full
audio cache therefore has complete source/input and stored-digest verification,
but not a full fresh-encoder replay.

The real feature-transport audit passed on 21 events / 63 questions, with 945
independent JSON decisions and seven exact fresh-process payload pairs. The
8-update-per-arm learning smoke also passed replay and accounting, but was
explicitly marked `quality_evaluated=false`; it is not a quality comparison.

## Existing long-fit result, kept separate

The already completed post-attention-normalization run is documented on the
separate [`codex/r2-av-postnorm-stability` branch](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-av-postnorm-stability/docs/pretrained_reuse/R2_AV_POSTNORM_LONG_FIT_20261017.md).
Its source commit was `afe03b78a1d6f997068623a9be6b6cce4ac97386`; its result
receipt SHA-256 is
`2b0c300ba650e3317000a579154c4ad05d1f8320c4dda2740b2a09b21d9a0825` and its
effective-config SHA-256 is
`998751914363c623736f418dd2d4d9d54de6b195cab4318f6fd5015`.

This was an `E0G0` seed-17, 32,768-update candidate run with an opt-in shared
LayerNorm after AV attention. It started from the fitted control8192 tensors
with a fresh optimizer; it was not a fresh-pretrained initialization and not an
optimizer resume. The LayerNorm added 384 parameters (197,098,333 total before
quantization). On the separate 8,875-example `validation-video` view, Accuracy
was 38.45% at step 512, 46.35% at 2,048, 60.42% at the selector-chosen step
16,384, and 61.84% at final step 32,768. Video NLL was 1.0865 at the selected
step and degraded to 1.2687 at final. On `validation-common`, final macro
Accuracy was 46.00% and macro NLL 1.6794 versus initial 37.42% and 1.6302.
Text Accuracy remained 24.85%; audio, image, text and video did not all reach
the product threshold.

This run is evidence that the fixed video validation view continued improving
past 512 updates under that run, with later calibration/NLL trade-offs. It does
not isolate LayerNorm's causal effect: the no-norm run stopped after 5,385
history records. It is not a learned conditional gate or bidirectional AV
exchange comparison. The candidate and validation remain research-only; no
quantization, audit evaluation or product promotion occurred.

## Decision and next gate

These checks close the complete frozen video-cache execution, expose a corrected
but still incomplete audio producer record, and confirm that actual event-clock
feature transport was exercised. They do not establish a qualified joint-AV
task: the admitted Greatest Hits labels do not prove both-required semantics,
precise physical synchronization, or actor/site independence. The 8,875-video
validation view has been repeatedly evaluated and used for checkpoint selection;
it is not a blind audit.

Keep the current Attention control and gated/exchange candidates unchanged. The
small gate is not rejected for parameter cost, but there is no admitted task
evidence here that warrants another gate-quality run. Before that run, establish
a genuinely complementary task with independent source groups and valid audio
swap, video swap, missing-input, timing-offset and contradiction labels. Define
the deployment device and task-specific confidence/latency criteria before any
release decision. Keep the <=200M figure as a pre-quantization parameter target;
the <=200MB package target is still unmeasured for this candidate.
