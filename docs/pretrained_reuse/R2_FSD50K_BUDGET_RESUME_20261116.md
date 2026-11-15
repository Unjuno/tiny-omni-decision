# FSD50K budget resume and C: storage check — 2026-11-16

## Verified attempt-08 source prefix

The preserved attempt-08 source at
`C:\CodexArtifacts\pretrained-reuse-r2\fsd50k-annotation-presence-v1-media-20261115-attempt-08`
was revalidated before considering a resume. The verifier checked the pinned
preflight manifest and ordered prefix, receipt hash, exact file set, all
17,590 WAV sizes, SHA-256 digests, and ZIP CRCs. It passed with 17,590 files:
12,845 train, 2,053 calibration, and 2,692 of 3,981 validation. There are 1,289
validation files still missing. Official eval labels and media remain unused.

The run stopped because its transfer counter reached 8,867,161,148 bytes,
116,484 bytes below the original 8,867,277,632-byte plan, and the next range
request was correctly rejected. Prior attempts outside attempt-08's resume
chain add 61,146,900 requested bytes: attempts 01 and 02, the non-chain portion
of attempt 05, and the non-chain portion of attempt 07. They must remain charged
if resuming from attempt-08.

## Resume guard implementation

Commit `eb3c7cbe29b6fa8124fb9ea21c377aa0052e9ab2` adds fail-closed recognition
for only the pinned attempt-08 runner hash and exact budget-exhaustion error.
It records the planned transfer budget separately from a new attempt's total
budget. An explicit retry buffer is required for resume and is capped at
1,500,000,000 bytes; the effective total cannot exceed the pinned source archive
sizes. The previous attempts' bytes remain accounted. The intended attempt-10
ceiling was 10,367,277,632 bytes, with 1,438,969,584 bytes left after charging
the attempt-08 chain and the 61,146,900 non-chain bytes.

Focused tests for acquisition, media validation, feature extraction, Decision,
and manifest handling passed (29 tests); Ruff and `git diff --check` passed.
GitHub Actions run 4587 for PR #108 passed lint, test, and manifest validation.

## C: versus D: output location correction

`C:\CodexArtifacts\pretrained-reuse-r2` is a junction to
`D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2`. An initial
attempt-09 copy mistakenly used that junction. The copy was stopped at 2,568
train WAVs; the partial directory is preserved on D: with no receipt or result
record and is not eligible as a resume source. No range request had been made:
the runner copies and verifies the prefix before reading the archive index.

Attempt-10 was redirected to the real C: path
`C:\CodexArtifacts\pretrained-reuse-r2-c-local\...`, but it was stopped during
the required read-only verification of attempt-08, before creating its output
directory. No acquisition is currently running. The user had previously asked
to disconnect D:, so further reads from that drive are paused pending direction
on whether to keep it connected long enough to copy the verified corpus to C:.

No attempt-08 media, manifest, checkpoint, feature, or label was modified or
deleted. No GPU work or model training occurred in this follow-up.
