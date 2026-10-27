# GRID two-slot correspondence development catalog

After the separate [one-slot attempt failed](R2_GRID_HARD_CORRESPONDENCE_ATTEMPT_20261028.md),
freeze this new task before model/media access. Use the same pinned provisional
train and development-validation partitions and same four-query balanced quartet
construction, but choose three same-speaker candidates at exactly **two** of
six different code positions from the anchor. Rank by SHA-256 of
`grid-two-slot-v1|<anchor>|<candidate>` and deduplicate canonical quartets.
Candidate IDs and quartet IDs use the prior fixed domains, and all four
candidate audio roles are emitted once per quartet. Keep every eligible source
row, count missing-neighbor anchors, and retain exact source hashes. Verify
same-speaker and split separation and exactly two changed slots from each
quartet's saved anchor. No media, model, prediction or sealed audit is read.

This is a two-slot development dataset whose difficulty differs from the
unavailable one-slot proposal and previous Lombard candidate policies. Four
roles per quartet control a deterministic video-only positional shortcut, but
do not prove full multimodal complementarity. Media availability, labels,
cross-corpus person identity, actual timing, matched unimodal baselines and
intervention controls remain required before a quality claim.
