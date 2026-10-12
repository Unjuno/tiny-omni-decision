# R2-A — Audio Decision model (Omni roadmap subsection)

**Position:** a subordinate development track within [ROADMAP.md](../../ROADMAP.md), primarily R2 (pretrained-feature reuse), with links to R4 (conditional alignment), R5 (observation reuse), R7 (deployment) and R8 (independent evaluation). The user requested this organization on 2026-10-12. This document does not replace the main Omni objective or rewrite historical experiment results.

## Goal and boundary

Build a small, locally executable **audio-conditioned Decision model** that accepts an audio observation, a question or decision criterion, and supplied options; returns calibrated option probabilities; and reuses the same encoded observation for multiple questions. Prefer already learned audio representations and minimal additional training.

Audio-only inference may be a separately evaluated component or development demonstrator. Its completion is **not** completion of Tiny Omni Decision. The parent product still requires text, image, audio and video, general condition following, at most **200M total resident parameters before quantization**, and the parent's >=95% per-modality quality target, aspiring to >=98%, on predeclared representative tasks. Narrow keyword accuracy cannot satisfy the general audio or Omni quality gate.

The audio track does not receive its own additional 200M allowance. Count any audio encoder, language encoder, ASR component, Lens, adapter and head in the deduplicated full-system total. Keep the existing quantized-package target and resource approvals; do not reject an FP32 reference solely because quantized packaging has not yet been measured.

## Separate capabilities rather than one pooled audio score

| Capability | Intended Decision behavior | Scope guard |
| --- | --- | --- |
| Keyword/short-command recognition | Identify a word or answer a condition about a known command | Useful component test; not general speech comprehension |
| Environmental sound decisions | Determine which event occurred and answer questions about presence, order or duration | Report event balance, missed positives and room/domain transfer |
| Speech-content and conditional decisions | Use what was said to answer supplied criteria, including changed questions and candidates | A transcript retrieval score or ASR accuracy alone is insufficient |
| Paralinguistic information, when required | Use nonverbal acoustic evidence that is relevant to the declared task | Add only with task-appropriate data, ground truth and evaluation; not an assumed existing capability |

These are capability slices, not a mandate to start four new training runs. Freeze the intended slice and its evaluation before allocating compute.

## Existing evidence and its limits

The following are repository-recorded experiments, not reruns performed by this documentation change. Evidence was read at commit `169f940a2ee6e1bbd355a39fd86b8045d1e87da1`.

- [Speech Commands and multi-query probes](R2_PATH_A_PROBES.md): pretrained Whisper Tiny features plus frozen MiniLM and a small readout supported narrow English keyword and derived-question tests. The reported four-question macro Accuracy of 97.85% is development evidence on 640 audio clips; three question types are mechanically derived from keyword labels. It is not a general audio reasoning score. The runtime replay's batch-size-8 agreement does not establish batch-size-1 service latency or identical BF16 behavior at arbitrary batch shapes.
- [Audio feature-dependence control](AUDIO_MULTIQUERY_FEATURE_DEPENDENCE_20261011.md): zeroed and other-clip features degraded the fixed readout. This supports use of audio evidence on that narrow task, not independent product qualification.
- [LibriSpeech transfer comparison](LIBRISPEECH_HARD_NEGATIVE_TRANSFER_20261011.md): the unchanged keyword-trained runtime remained near the four-choice chance reference on transcript queries. Retain this failure as a transfer limitation rather than extrapolating the keyword result.
- [STARSS23 comparison](R2_STARSS23_COMPARISON_20261011.md): a synchronized audio/video development comparison was executed. Audio-only outperformed the tested joint readout; the training-prevalence baseline was already strong, several events had very few positives, and validation covered only two rooms. This is a reason to keep an audio-only baseline, not proof that audiovisual fusion cannot help.

Historical metrics, prediction files, checkpoints, failed attempts and dataset splits remain unchanged. Keep results for different corpora and question types separate.

## Minimal implementation and learning approach

1. Reuse a verified, pinned pretrained audio encoder. Treat Whisper-derived speech features and environmental-audio features as alternatives or separately counted components; neither is automatically suitable for all audio tasks. Preserve the strongest existing audio-only checkpoint as the baseline.
2. Separate **observation encoding** from **question/candidate scoring**. Cache question-independent audio features with the audio hash, model and preprocessing revisions, sampling/channel policy, time bounds, masks, dtype and tensor shape. Changing the question must not silently require re-encoding the same observation.
3. Train only the smallest justified conditional readout first. Start from audited labels; optional Teacher supervision must have matching option identities/order and a demonstrated benefit. Do not make a Gemma Teacher, full embedding imitation, a new world-model pretraining run or weight merging mandatory.
4. Check semantic behavior as well as average metrics: change the criterion while keeping audio fixed, reorder candidates, replace or remove audio, and test unseen recordings/speakers/rooms and question templates. Reusing a feature for four deterministic keyword-derived labels does not establish general condition following.
5. Diagnose failures before enlarging the model. Compare temporal/token-level features against pooling, or a small adapter/selective update only when the error evidence justifies it. An input-side Lens still requires gradients through a frozen backbone; frozen weights do not eliminate that activation cost.
6. After the chosen audio task is viable, compare supported deployment precisions and actual kernels. Measure raw-audio-to-decision behavior, cache-hit queries, batch-one and batched paths separately. Keep unsupported export/low-bit paths explicitly unverified.

## Audio-to-Omni integration contract

The audio component must expose reusable feature tokens/state in addition to its optional audio-only probabilities. Preserve event/time metadata and missingness so that fusion can use information not recoverable from a narrow class score. A transcript may be an explicitly evaluated auxiliary path, but must not silently replace all acoustic evidence.

Use the parent observation identity, typed decision/options, cache invalidation, calibration and provenance contracts. Reuse shared language/decision modules where functional equivalence is verified; do not duplicate them unnecessarily or assume sharing reduces operation count.

Integration is a separate experiment: compare matched audio-only, video-only and joint inputs, including mismatched/absent-modality controls, on task-matched synchronized data. Improvements must be judged on relevant held-out decisions and their uncertainty, not merely on the existence of a joint forward pass. An audio-only win must remain visible when joint fusion performs worse.

## Subtrack milestones and acceptance

| Milestone | Required evidence | What it does not establish |
| --- | --- | --- |
| Reproducible audio baseline | Pinned source/model/data, finite forward, saved readout and verified reload, scoped task results | Broad speech or Omni capability |
| Reusable audio Decision interface | Same observation serves changed questions/options; cache and permutation checks pass | Unseen-condition generalization by itself |
| Task-quality gate | Independently grouped evaluation; Accuracy, balanced/macro metrics, NLL, Brier, calibration/abstention and uncertainty; failures disclosed | Success on unrelated audio distributions |
| Local deployment gate | Actual deduplicated parameters, complete package bytes, peak resident memory, raw-input and cache-hit latency distributions | A full Omni device result |
| Omni integration gate | Evidence the joint system uses synchronized audio where needed, with unimodal and shuffled controls, inside the parent's total budget | Completion of other untested modalities |

Keep speaker/recording/room groups disjoint as appropriate, distinguish development validation from untouched final evaluation, and report minority-event counts. Fix task-specific confidence and resource criteria before promotion. `PASS`, `FAIL` and `UNCERTAIN` apply per milestone and capability; a narrow passing subsection does not turn the parent roadmap into a completed project.

## Work allocation and change scope

Track audio progress separately from video/world-representation reuse, image/text quality, shared-backbone/merging research and final multimodal integration. Reuse the audio component's proven code, caches and evaluation tools in Omni, without allowing easy audio metrics to obscure the unresolved main-model requirements.

This is a documentation-only organization change. It starts no training/download/paid compute, changes no dependencies or source code, and does not stop a process, alter a legacy experiment, close a PR or merge into `main`. Existing permissions for bounded local experiments are unchanged; this subsection does not expand them.
