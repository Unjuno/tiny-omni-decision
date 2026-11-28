# IntPhys train archive resumable download guard (2026-11-28)

## Why this was added

The live `train.2.tar.gz` transfer used curl retry with `--output` but without
`--continue-at -`. The saved partial was observed falling from 17,966,592,000
bytes to 14,020,608 bytes after a retry. This is consistent with the retry
reopening the output in truncate mode. The active process was not changed by
this code increment.

## Implementation

`scripts/download_intphys_train_archive.py` uses the same four byte-count and
MD5 pins as the feature extractor. The pins now live in the lightweight
`scripts/intphys_train_archive_pins.py` module so download and extraction code
share one source of truth.

Before appending to an existing partial, the downloader requires the supplied
strong ETag, a `206` response, an exact `Content-Range` start matching the
local byte count, and the pinned total length. It sends `If-Match` on every
request, preserves partial bytes on transient failures, retries from the new
local length, and fails closed if the server returns a full `200` during a
resume. It checks available disk space against remaining bytes plus a 20 GiB
default reserve, then verifies full byte count and pinned MD5 before returning
SHA-256 metadata.

Example invocation after the active transfer has been safely stopped and the
caller has confirmed the source ETag:

```powershell
python -m scripts.download_intphys_train_archive `
  --shard-number 2 `
  --etag '"5d9bc78d-8a3fd7c78"' `
  --output 'C:\CodexArtifacts\pretrained-reuse-r2-local\intphys-full-train-coverage-20261128-attempt-01\downloads\train.2.tar.gz.partial'
```

This helper has **not** been applied to the currently running curl process.
No archive was downloaded or hashed by the helper in this increment.

## Verification

CPU-only tests simulate a stream interruption and verify that the retry starts
at the exact saved byte offset. They also prove that an unexpected status,
ETag, or range leaves the partial unchanged. Focused IntPhys downloader,
feature-extraction, and shard-merge tests passed (26 tests); Ruff and
`git diff --check` passed. No GPU or network test was run.
