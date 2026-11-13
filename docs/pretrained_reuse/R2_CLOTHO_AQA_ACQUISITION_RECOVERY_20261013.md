# Acquisition recovery after verified terminal HTTP504

Attempt01 terminated with `HTTP Error504: Gateway Time-out` in
`HttpRangeReader.read -> urllib.request.urlopen`. Its process69416 disappeared
and exec session67274 returned exit1; result.json is terminalFAIL with205 saved
records and320,767,709 transferred bytes. No intervention in a live process.
The retry option covered truncated response bodies but did not catch HTTP errors.
The upstream cause of the gateway timeout is unknown.

Added three attempts for explicit502/503/504 responses with1/2s delays, preserving
the same requested range/position, TLS checks and existing byte budget. Forbidden
responses such as403 and invalid200/Content-Range are not retried. Existing
short-body retries remain independently bounded at three attempts. The retry
wrapper does not weaken media checks or treat unfinished acquisition as complete.

Verified terminal-attempt reuse requires identical frozen split hashes, allowed
source IDs, matching split/uploader/license/member and every saved file's SHA
and length. Current archive CRC/length checks still run when the copied files
are consumed. All205 records passed. Source125825/156653 exposed an existing
name-check inconsistency: archive names retain`&#39;` but acquisition lookup
decodes it. Applied the same HTML decoding to provenance gates, retaining the
literal member name and rejecting different identities. No fuzzy matching.

New run source3f7482e:

`python -m scripts.fetch_clotho_development_media --reuse C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-development-media-01 --output C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-development-media-02`

PID15788 confirmed live; new output reached207/856 at observation. Attempt01 is
preserved. All questions, labels, seed, split hashes,856 source IDs,
feature recipes and model comparison conditions are unchanged. Downstream
decode/screen/comparison defaults now point to attempt02; full feature command
must use `--media .../clotho-aqa-development-media-02`. Existing frozen caches
and learned question control remain intact. No full audio training occurred.

Targeted recovery/source/integrity tests passed, including exact range retry,
attempt bound,403/invalid-range refusal, corrupt saved bytes, active-run refusal
and HTML name parity. Repository-wide Ruff passed;790 CPU tests passed,1skipped.
New acquisition must terminatePASS and pass
all downstream media checks before learning. Its atomic result.json and live
session9957 are authoritative; this launch record is not completion.
