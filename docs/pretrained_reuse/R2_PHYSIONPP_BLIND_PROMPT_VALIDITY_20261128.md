# Physion++ blind review prompt validity check — 2026-11-28

## Finding

A read-only, label-blind check of cards 011–020 in the earlier AI-oriented
review packet found that its fixed wording — “red object” and “yellow-marked
target” — does not consistently match the visible objects and target-area
colors. Several cards show green, blue, purple, or dark objects and differently
colored regions; only some visibly match the wording. The instruction is
therefore not a valid universal anchor for this packet.

This is a prompt/interface issue only. It does **not** show that the official
contact labels are wrong or that the events are visually unobservable. No
left/right outcome judgments or confidence scores were recorded, so this is
not an accuracy result. Review stopped at ten cards rather than generating
answers from a mismatched instruction.

## Integrity boundary

- Packet: `physionpp-blind-observability-review-20261128-attempt-01`.
- The packet's `README.md` SHA-256 is
  `4453e835b50edc9f7a6f928a7440c4e6266d3aba5381992719e544d5f0268448`;
  `packet-manifest.json` SHA-256 is
  `a5943bf677151729b1be8cded68d8171fb6a6fac93628d37e908203d7283535f`.
- The original blank `responses.csv` still has SHA-256
  `882432f4397fcae5b2176e7b31a096e354d7342acae4c5b643e55e1e9d8b4981`.
- Cards 011–020 were viewed in ID order. No answer key, source labels, scenario
  names, official metadata rows, validation/test media, or sealed data were
  opened. No packet, response file, media, labels, model, or checkpoint was
  changed.
- This is an AI-assisted prompt check, not a human review. The newer
  reviewer-facing pack at `physionpp-blinded-review-20261128-attempt-02` is a
  separate artifact; this check does not invalidate or mark that human review
  complete.

## Next data step

Before any reviewer records judgments, use scenario-neutral task wording based
on the official `target_contacting_zone` event, not a fixed object/zone color.
If the intended target cannot be identified from the rendered clip, reviewers
must be able to mark the item uncertain. Keep side assignment and answer key
blind until completed judgments are frozen. Then check reviewer agreement and
report ambiguous/occluded cases without relabeling the corpus.

The trained model and Path A result should remain unchanged until this data
review is complete. Existing train-only factor and temporal-sampling findings
remain diagnostics; they do not establish model quality or a task ceiling.
