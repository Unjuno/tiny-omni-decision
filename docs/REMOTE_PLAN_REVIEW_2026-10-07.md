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
and probabilities are material for a frozen reference/cache contract.

The local environment confirms PEFT 0.21.2, 205 LoRA layers with FP32 adapter
tensors, and BF16 base tensors. In PEFT's unmerged forward path, it computes the
LoRA branch using the adapter dtype, adds it to the base-layer output, then
casts the result to the base output dtype. The safe merge path instead casts
each delta to the base weight dtype before adding it in-place to the base
weight. This establishes a real arithmetic-order/dtype difference, though it
does not by itself prove that this is the only source of every changed
prediction.

To test whether separate BF16 rounding caused the mismatch, a second in-memory
merge computed `FP32(base) + FP32(delta)` per layer, then cast the merged matrix
once back to its original BF16 dtype. It used the same 2,048 validation IDs and
order, with no training or disk checkpoint. It still changed 13/2,048 top-1
choices: Text 4/512, Image 5/512, Audio 0/512, Video 4/512. Maximum logit and
probability deltas were 1.75 and 0.278382. The single-rounding merge therefore
did not restore adapter equivalence; it changed which records flipped. Macro
Accuracy was unchanged at displayed precision (0.779785 vs 0.779785), while
macro NLL/Brier/ECE shifted from 0.547123/0.269738/0.055503 to
0.546473/0.269509/0.054624. This does not establish a parity guarantee.
Differences remained concentrated in Image/Text confidence, with probability
deltas over 0.1 on 6 image/text examples; Audio probability deltas remained at
or below 0.007563 and no Audio choices changed. Full predictions and metrics
are retained in the local artifact paths below.

This is not evidence of corrupted weights. It is an execution-path difference
between adding a low-rank activation update and folding that update into a
BF16 dense matrix. The merge gate remains unresolved for a reference/cache
contract; do not claim exact equivalence or silently relax the tolerance after
seeing these results. A future Q1 comparison must explicitly declare whether
it measures ternary damage relative to the adapter execution path or to a
frozen merged BF16 reference, and report merge deviation separately.

Local diagnostic artifacts (not committed):

- Safe merge: `C:\Users\junny\AppData\Local\CodexArtifacts\teacher-quality-reloads\clean-dev-v2-step1536-merge-equivalence-20261007\`
- FP32-accumulation merge: `C:\Users\junny\AppData\Local\CodexArtifacts\teacher-quality-reloads\clean-dev-v2-step1536-merge-fp32accum-20261007\`
- FP32-accumulation prediction SHA-256:
  `969e36d6095feeb43a18a8bd52341d2ee80f4d0c96b3c8b3fba715c16993d955`
- FP32-accumulation metrics SHA-256:
  `70b8c42400e39aca7f9f63297b5993a1aa23bdb63180fc6421bad459fc224345`

## Next steps

1. Keep the completed E-long run and both evaluation outputs immutable.
2. Treat exact adapter/merged parity as unproven. If the research experiment
   proceeds, preregister the Q0 execution path, merge tolerance, and separate
   top-1/probability deviation report before examining Q1 outcomes.
3. The EmbeddingGemma 2 model card identifies an all-modality embedding model
   with 740M parameters and a 768D output ([official model card](https://huggingface.co/google/embeddinggemma-2)).
   The local environment has Transformers 5.6.2, which does not recognize its
   `embedding_gemma2` config; `sentence-transformers` is not installed. Only
   small metadata files were fetched at immutable revision
   `914f7f89142e33e77833254d9c9b90c3cef7303b`; no model weights were downloaded
   and no working dependencies were upgraded. A separate pinned runtime is
   required before the student gate can run.
4. Reconcile the product student's option-scoring/calibration contract with
   the teacher-cache and quantization plans. An embedding vector supports all
   four media types, but the project still needs an evaluated option scorer
   that returns calibrated probabilities over the supplied choices.
5. Preserve the proposed 1,024-update Q2/Q3/Q4 budgets if still feasible after
   the student/Q0 gates; no extra E-long steps or seeds are justified by this
   plan review.

This note records plan reconciliation and a failed engineering gate. It does
not promote the E-long checkpoint, claim compression, or alter the canonical
four-modality quality objective.
