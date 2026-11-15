# R0 live audit — 2026-11-15

Read-only snapshot from the `codex/pretrained-reuse-active-20261114` worktree at 2026-11-15 21:48 JST.

## Git and pull requests

- `origin/main`: `09246d7b1821662a6b1eba47fc08c43f9dc3863c`.
- Roadmap PR #15 is Open / Draft / not merged, based on that main revision; its head is `bfb13bf6e9ac4f2ce244eb51e470afa907587380`.
- GitHub Actions run #4224 (`43361862245`) for PR #15's head completed successfully. This is evidence for that PR head only.
- The active progress branch is `codex/pretrained-reuse-active-20261114`, at `5154729a` and equal to its origin tracking ref after fetch. No open PR for this branch was found. GitHub returned no combined status entries for this docs-only head; do not treat that as a green check.
- Local `gh` CLI is unauthenticated; PR/check evidence above came from the connected GitHub read tools. No PR or branch was changed.

## Live resources and active run

- FSD50K development-media acquisition PID 5452 is still running the frozen attempt-06 command. At the observation, `progress.json` reports 13,400 / 18,879 durable files; recursive WAV inventory reports 13,411, so 11 files are ahead of the last durable checkpoint. There is no `result.json`. The download remains partial and unverified; official eval labels/media remain unopened.
- RTX 3080 Laptop reported 8,270 / 16,384 MiB used and 99% utilization. Do not launch competing GPU work while this separate process remains active.
- C: had 258.94 GiB free. The active attempt's persisted cumulative range count is 6,856,749,442 bytes against an 8,867,277,632-byte budget; this state predates the 11 uncommitted-to-progress files and is therefore only a lower-bound snapshot.

## Frozen FSD50K manifest

The local `preflight.json` manifest hash is `0bbe5db5e6167a4c041e349889920f8da0f1a0703ee00fa63ff12a59a0531b6d`. A separate read-only canonical rehash verified all three JSONL counts, ordered sample-ID hashes, and full ordered-row hashes, with zero pairwise sample-ID, source-ID, and uploader-group overlap. This verifies the manifest bytes and group split, not downloaded-media integrity or task quality. See `R2_FSD50K_ANNOTATION_DECISION_PREFLIGHT_20261115.md` and `R2_FSD50K_ACQUISITION_VERIFICATION_20261115.md`.

## Open gates

- Validate the complete acquired media against the frozen manifest, receipt, hashes, and archive CRC before feature extraction.
- Measure the narrow annotation-presence task on held-out uploader groups; it is not broad audio understanding, and absent annotation is not verified acoustic absence.
- R2 modality quality and both-input-required multimodal evidence remain open; do not start shared-Lens/merge work without a measurable R1/R2 justification.
- R0 product device latency/energy budget and complete approved task battery remain open; engineering reference targets are not release evidence.
