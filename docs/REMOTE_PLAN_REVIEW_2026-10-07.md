# Remote plan review — 2026-10-07

## Remote state inspected

- `origin/main` advanced to `b847f47` (EmbeddingGemma 2 deployment-student path).
- `origin/codex/roadmap-compression-video-specialization` is at
  `91d7867` (reviewed compression-first / post-quantization experiment plan).
- The long-budget branch remains `ca981af` before this report; its CI passed
  (GitHub Actions run `37515151693`, job `112446052354`).
- `origin/main` CI at inspection passed (run `37514258468`, job
  `112443001276`).

## Plan reconciliation

The updated `main` now places an EmbeddingGemma 2 high-precision decision
student gate between the Gemma 4 Teacher and ternary conversion. The separate
compression-first plan instead defines Q0 as a merged Gemma 4 Master and Q1 as
that Master ternarized. These are different quantization-source architectures.
The two plans should not be silently combined.

For the product path, follow the current `main` order: freeze the Gemma 4
Teacher as an external quality reference, inspect and pin EmbeddingGemma 2,
evaluate/adapt its high-precision student on the same eligible held-out IDs,
then quantize only the selected student. The Q0–Q4 recipe can still inform a
later controlled comparison, but its base must be reconciled with this student
gate before a training manifest is frozen. The revised 95% requirement is a
final shipping gate; compression-entry evidence is a distinct research gate.

## Q0 merge-equivalence check

The selected E-long step-1,536 adapter was independently reloaded before this
check; that reload reproduced all saved predictions exactly. A second local
evaluation applied `PeftModel.merge_and_unload(safe_merge=True)` in memory and
used the same 2,048 ordered E validation IDs. No training or sealed-audit
access occurred, and no merged weights were written.

| Metric | Adapter reload | In-memory merge |
|---|---:|---:|
| Macro Accuracy | 0.779785 | 0.779297 |
| Text Accuracy | 0.810547 | 0.810547 |
| Image Accuracy | 0.832031 | 0.830078 |
| Audio Accuracy | 1.000000 | 1.000000 |
| Video Accuracy | 0.476562 | 0.476562 |
| Changed argmax predictions | — | 14 / 2,048 |
| Maximum option-logit delta | — | 1.125 |
| Maximum option-probability delta | — | 0.255914 |

The strict equivalence check failed. Changes by modality were Text 4/512,
Image 5/512, Audio 0/512, and Video 5/512. The worst logit delta was on a
Clevr-4 image example. Aggregate scores remain close, but the altered choices
and probabilities are material for a frozen reference/cache contract. The
precise numerical mechanism has not been isolated; BF16 merge rounding is a
candidate explanation, not an established cause. Do not pass this result off as
equivalent or proceed to a ternary comparison that assumes equivalence.

## Next steps

1. Keep the completed E-long run and both evaluation outputs immutable.
2. Diagnose merge arithmetic/dtype and verify the comparison uses identical
   loaded base, adapter, processor, and input tensors. Set a predeclared
   tolerance based on numerical behavior while separately reporting every
   top-1 change; do not weaken a threshold after seeing outcomes.
3. In parallel, implement only the EmbeddingGemma 2 model/processor inspection
   and held-out input compatibility gate. Do not begin training until the
   repository's two quantization-source plans are reconciled into a versioned
   run manifest.
4. Preserve the proposed 1,024-update Q2/Q3/Q4 budgets if still feasible after
   those gates; no extra E-long steps or seeds are justified by this plan review.

This note records plan reconciliation and a failed engineering gate. It does
not promote the E-long checkpoint, claim compression, or alter the canonical
four-modality quality objective.
