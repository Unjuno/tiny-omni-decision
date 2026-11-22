# R0 live audit — 2026-11-23

## Snapshot

- Repository: `Unjuno/tiny-omni-decision`.
- Canonical roadmap branch observed at `ea014431b163f69176c52f5df8944f2dc7f7583f`. Its PR #15 remains open, draft, mergeable, based on main `09246d7b1821662a6b1eba47fc08c43f9dc3863c`. Both reported PR #15 test checks succeeded on that head.
- BERT-family evidence is now proposed separately in draft PR #114, based on the canonical roadmap branch at the commit above. Head is `dae1ee505ef2c110f6aa47b66f19a7c2c41fcff0`; all three reported GitHub Actions test checks succeeded. This remains development-only text evidence and is not an Omni quality pass.
- The user's primary workspace remains on `codex/teacher-quality-clean-dev-v1` with pre-existing uncommitted changes in `ROADMAP.md`, `docs/VIDEO_TEACHER_V2_LONG_BUDGET.md`, `design-qa.md`, `docs/pretrained_reuse/`, and related scripts/tests. It was not edited, staged, stashed, reset, or cleaned.
- This snapshot was prepared in a separate clean worktree from the canonical roadmap head. No legacy run, checkpoint, corpus, media payload, official test, or sealed audit was opened or changed.

## Local compute and storage

- At 08:47 JST, C: had 152,179,273,728 bytes free (about 141.7 GiB).
- RTX 3080 Laptop GPU: 8,259 / 16,384 MiB allocated, 48% utilization, 58 C. PID `138728`, started at 08:20:04 JST, belongs to Orbitune's existing `compound_cfe_train.py` run on the user's existing checkpoint. It remains untouched. No GPU work was started for this audit.

## BERT-family finding

The current evidence supports retaining Japanese BERT-base as a text teacher/reference, not as the Japanese tower in the four-tower candidate: the recorded replacement estimate is about 253.2M pre-quantization parameters, over the provisional 200M total. Grouped JCoLA adaptation lifted minority-class recall but reduced sample accuracy and worsened calibration. The compact BERT-derived 27.6M encoder passed a CPU load/runtime smoke but has not passed a new independent Decision-quality test. PR #114 records these distinctions and does not claim multimodal quality.

## R2 source/task and quality gates

- A metadata-only recheck of CUHK-X and EgoInertia-MI found that both had already been screened on the canonical roadmap branch; no duplicate source-screen report was created.
- CUHK-X still requires a non-commercial research Data Use Agreement, is not redistributable, and the challenge release excludes RGB. The official page advertises cross-subject recognition and action-understanding/reasoning tasks, but the existing project screen already marks this source `REVIEW_NOT_ADMITTED`. No DUA was accepted and no payload was downloaded.
- EgoInertia-MI still requires an author-reviewed access request with name, affiliation, and intended research use. It is already `REVIEW_NOT_ADMITTED`; no request was submitted and no data were accessed.
- R2 M-0/M-1 remain partial. R2 M-2 has no admitted, rights-cleared task that proves a genuine joint-input Decision improvement over the best unimodal comparator with acceptable calibration. Exposed validation and sealed audit remain out of scope for further tuning.
- R0's complete task battery and target-device latency/energy criteria remain open. R1's exact V-JEPA 2.1 weight grant remains under review. R3+ architecture selection and quantization are not justified by the present evidence.

## Decision

The requested local multimodal Omni Decision model is not complete. Keep BERT as a text teacher/reference; do not launch another BERT run on already observed splits. Continue R2 task admission and implementation checks only on sources with clear rights, auditable independent groups, and a target that needs the proposed input streams. Preserve the live Orbitune process and all existing project artifacts.

