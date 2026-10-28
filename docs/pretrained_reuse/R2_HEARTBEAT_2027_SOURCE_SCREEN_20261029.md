# HEARTBEAT Challenge 2027 source/task screen — 2026-10-29

## Decision

`REVIEW_NOT_ADMITTED`. This is the closest newly released task format found so
far to the product's typed Decision objective: select a response action from
four options as JSON, and make online audio-visual decisions about when an
event warrants a response. The task fit is strong, but the inspected sources do
not separately establish dataset/media rights, consent, or independent
speaker/conversation splits. The data are also too large for current local
storage. Nothing was downloaded or opened.

## Task fit

The [official challenge page](https://heartbeat-challenge.github.io/) describes
Track I as choosing among `WAIT`, `RESPOND`, `CONTINUE`, or `YIELD` from an
audio prefix and assistant instructions, with JSON output and Top-1 scoring.
Track II evaluates streaming proactive understanding on separate audio-visual
and audio-only tracks, with a JSON output for every chunk, including chunks
that do not trigger a response. This naturally exercises action selection,
timing, and abstention/silence.

The limitations matter: Track I is audio plus text instructions, not a
text-free audio task. Track II is an online event-response task, not a
fixed-option calibrated-probability benchmark. The official track descriptions
do not by themselves demonstrate that each labeled AV event requires both
modalities; matched audio-only, video-only, replacement, and timing controls
would still be needed.

## Pinned release and size evidence

The official [baseline repository](https://github.com/heartbeat-challenge/HEARTBEAT_challenge_baseline/tree/53cf8aa4d55ed522facd8083d56bc998668dc847)
was inspected at commit `53cf8aa4d55ed522facd8083d56bc998668dc847`. Its Apache-
2.0 license covers the baseline repository code. The linked [data/model
repository](https://huggingface.co/inclusionAI/Realtime-Venus/tree/a1d95d3ac61416c8db34736b152241396850a6ed/data/HEARTBEAT_Challenge)
was inspected at revision `a1d95d3ac61416c8db34736b152241396850a6ed`; the
pack index identifies data commit `007097ab5d304ff1fff154bedf415b088717ec8d`.

The publisher lists 132,571 records and about 599.83 GB packed. Track II AV
has 30,000 training records (294.31 GB) and 1,000 development records in one
9,282,733,388-byte shard. The pack index gives that AV development shard
SHA-256 `2e2cd9087c99754fcfb6636b79bbf3636cfd32122f176ba4725c51c282013619`.
Track I audio turn-taking has 18,000 training records (11,116,885,630 bytes in
two pack files) and 1,000
development records (590,237,305 bytes; SHA-256
`83ebe3a0676fb84cf53fb482ff5d551f3ef34b75cc40573c2442c73343d7e483`).
These are public pack-index metadata hashes, not hashes of locally downloaded
files.

At the read-only resource check, C: had 778,063,872 bytes free. The AV dev
shard alone is about 12 times larger, and the 3.53 GB audio-only dev shard
also exceeds available free space before export or caching. No alternate local
volume was present. The baseline supports sample-row and HTTP-range reads, but
that does not resolve the data-use or split-integrity gates.

## Rights, groups, and disposition

The baseline code repository declares Apache-2.0 and the linked HF repository
card displays Apache-2.0, but the inspected challenge dataset README does not
state separate dataset terms, source-media provenance, or speaker-consent
terms. I do not infer a grant for underlying conversational recordings from a
code/model repository license. The challenge page asks participants to use
open-source data and models; it does not provide a separate dataset license in
the materials inspected here.

The public material describes train/dev partitions and source shards, but this
metadata-only screen did not establish participant-, speaker-, conversation-,
or source-level disjointness. It also did not identify a sealed final audit.
Therefore do not download, train, or evaluate on this corpus yet. Admission
requires explicit rights/consent terms, group-level split verification, a
task-specific audio/AV necessity protocol, and a storage-safe run plan.

## Reproducibility and limits

Metadata-only audit of the [challenge description](https://heartbeat-challenge.github.io/),
[pinned baseline](https://github.com/heartbeat-challenge/HEARTBEAT_challenge_baseline/tree/53cf8aa4d55ed522facd8083d56bc998668dc847),
and linked [dataset README/pack index](https://huggingface.co/inclusionAI/Realtime-Venus/tree/a1d95d3ac61416c8db34736b152241396850a6ed/data/HEARTBEAT_Challenge)
on 2026-10-29. No data pack, record, audio, video, or checkpoint was fetched.
No training, quality evaluation, or sealed-audit access occurred. The
[machine-readable screen](../../manifests/pretrained_reuse/heartbeat-2027-source-screen-20261029.json)
stores pinned revisions and metadata hashes.
