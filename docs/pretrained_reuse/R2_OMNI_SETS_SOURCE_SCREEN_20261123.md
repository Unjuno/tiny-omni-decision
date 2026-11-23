# R2 Omni-Sets source screen — 2026-11-23

## Disposition

**REVIEW_NOT_ADMITTED.** Omni-Sets is a plausible synchronized audio/video
candidate for a narrowly scoped cross-modal Decision task, but this screen does
not authorize access or use. No account information was shared, no access terms
were accepted, and no repository payload, annotation, media, model, validation
label, or sealed audit was opened.

The source is not ready for the product-quality experiment. The dataset card
describes its `omni` records as Qwen3-Omni-generated cross-modal conversations,
not human-adjudicated labels or supplied-option decisions. It claims every
pair needs both audio and video; that claim would need to be tested with
audio-only, video-only, wrong-stream, and temporal-misalignment controls. The
card's documented usage exposes a single `train` split, so a project-owned
video-group split and leakage audit would be needed before any evaluation.

## Official-source findings

Screened on 2026-11-23 from the official
[MBZUAI/Omni-Sets dataset card](https://huggingface.co/datasets/MBZUAI/Omni-Sets)
and the underlying
[HuggingFaceFV/FineVideo dataset card](https://huggingface.co/datasets/HuggingFaceFV/finevideo).

- Omni-Sets reports 40,087 Omni samples sourced from FineVideo. The sample
  identity is documented as `finevideo_{youtube_id}` and records contain
  video with an audio track, multi-turn Q&A, and optional synthesized spoken
  questions.
- The authors state that video and extracted audio are jointly passed to
  Qwen3-Omni and that the resulting Q&A requires both modalities. The labels
  are therefore generated supervision; no independent human answer adjudication
  or supplied-choice benchmark is documented in the card.
- The aggregate Omni-Sets annotation license is CC BY-SA 4.0. Its card says
  underlying media retain each upstream source's terms and specifically warns
  against relying on the aggregate license for commercial use.
- FineVideo describes its underlying videos as individually CC-BY YouTube
  videos, but access is gated. The access screen requires accepting its terms
  and sharing the account email and username with maintainers. Terms also
  require maintaining the current usable FineVideo version and passing those
  terms on when hosting or sharing FineVideo.
- FineVideo's data card reports 43,751 source videos and about 673 GB for the
  full dataset. Omni-Sets reports 936 GB aggregate size. No data was fetched;
  these figures are not a local-storage estimate for the Omni subset.
- The cards do not document an independent train/validation/test partition for
  the Omni subset. If access is later authorized, all records from the same
  YouTube/video identity must be kept in one split, and IDs/content must be
  checked against the project's existing exposure ledger before use.

## Gate decision

Do not accept the gated access terms or send account information on the user's
behalf. Do not use the CC BY-SA generated answers for product training without
a separate project-level license decision. Reconsider only after the user
authorizes the access gate and the project confirms the intended use can comply
with FineVideo's version/attribution/redistribution conditions. Even then,
admit only a source-disjoint video-grouped subset after auditing label quality
and the both-input controls; a split of this generated dataset would be
development evidence, not independent ground truth for the 95–98% objective.

## Effect on current goal

This candidate does not unblock R2-M. Continue searching for an accessible,
rights-cleared task with synchronized streams, independent groups, and a
decision target that can demonstrate genuine joint-input benefit. Keep the
existing text BERT results as text-reference evidence only. No model, corpus,
checkpoint, active run, or user worktree was changed by this screen.
