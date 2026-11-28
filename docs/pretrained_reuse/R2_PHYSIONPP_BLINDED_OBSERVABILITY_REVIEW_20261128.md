# Physion++ blinded label-observability review packet — 2026-11-28

## Purpose and status

The existing 75 exact-recorded-factor matched No/Yes pairs were displayed in
contact sheets with No always on the left and Yes always on the right. Those
sheets cannot support a blinded human review. This stage makes a separate
review packet from those same train-only frames, with pair order and side
assignment randomized, anonymous item IDs, and no labels, sample IDs, scenario
names, or factor metadata in the reviewer-facing files.

The response form is still blank. Therefore the data-observability gate remains
**OPEN**; no human review result is claimed. The packet uses 32 uniform
time-sampled thumbnails per clip at the existing 128×128 aspect-preserving
tile size. Reviewers can mark left, right, both, neither, or unclear, give
confidence, and note temporal ambiguity. This is a triage screen, not an expert
relabel or a replacement for a locked independent evaluation.

## Integrity

- Source: existing train-only pair manifest; SHA-256
  `a22fa791b05559dea1e2dcdd91dac687db07e48b9965b6a8fc82d955feb17293`.
- Cards: 75 PNGs, each 2048×536; filenames and public review IDs do not encode
  source IDs, pair IDs, scenario, factors, or labels.
- Public packet manifest SHA-256:
  `a5943bf677151729b1be8cded68d8171fb6a6fac93628d37e908203d7283535f`.
- Blank response CSV SHA-256:
  `882432f4397fcae5b2176e7b31a096e354d7342acae4c5b643e55e1e9d8b4981`.
- All 75 card hashes and dimensions verified; all 75 response rows are blank.
- The mapping and randomization seed are stored in a separate local answer-key
  directory and are excluded from Git and the reviewer packet. Do not open or
  share the answer key until responses are frozen.
- Reviewer packet size: 32,817,322 bytes. No media was newly downloaded or
  duplicated into Git; the packet reuses derived contact-sheet frames from the
  existing train audit.

## Reproduction

```powershell
python scripts/build_physionpp_blind_observability_packet.py `
  --pair-manifest C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-train-data-audit-20261128-attempt-06\visual-review-pairs.jsonl `
  --contact-sheet-dir C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-train-data-audit-20261128-attempt-06\contact-sheets `
  --packet-dir C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-blind-observability-review-20261128-attempt-01 `
  --private-key-dir C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-blind-observability-answer-key-20261128-attempt-01
```

- Builder: `scripts/build_physionpp_blind_observability_packet.py`
- Tests: `tests/test_build_physionpp_blind_observability_packet.py` — 3 passed
- Reviewer packet:
  `C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-blind-observability-review-20261128-attempt-01`
- Response form: `responses.csv`

The reviewer must fill `choice` (`left`, `right`, `both`, `neither`, or
`unclear`), `confidence_1_to_5`, and optional notes. Freeze and hash the
completed response file before matching answers. Only then may the separate
answer key be used to summarize correctness and uncertainty.
