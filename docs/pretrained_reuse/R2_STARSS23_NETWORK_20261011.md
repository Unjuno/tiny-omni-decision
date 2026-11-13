# STARSS23 acquisition transport diagnosis

The fixed 78-recording preflight was launched with
`scripts.fetch_starss23_frozen_media.py` into
`C:\CodexArtifacts\pretrained-reuse-r2\starss23-media-preflight-01`.
PID 65896 remained live while waiting for remote ZIP metadata. At the observed
time its HTTPS socket was IPv6 `SynSent`, with no plan/result file yet emitted.
This is not evidence that acquisition completed or that the dataset failed a
quality gate. No training process was involved.

On the same official LICENSE URL:

```powershell
curl.exe -4 -I --connect-timeout 10 --max-time 20 https://zenodo.org/api/records/7880637/files/LICENSE/content
curl.exe -6 -I --connect-timeout 5 --max-time 8 https://zenodo.org/api/records/7880637/files/LICENSE/content
```

IPv4 returned HTTP 200 in about two seconds; IPv6 exited with curl error 28
after about five seconds. DNS returned six IPv6 addresses. Sequential address
timeouts are a plausible explanation for the Python preflight delay, not a
proven ZIP-serving fault. A same-URL HEAD through the new Python IPv4 HTTPS
handler also returned 200.

`scripts/run_starss23_ipv4.py` wraps only the single-purpose acquisition child
process. It changes TCP address selection to IPv4 while retaining the original
HTTPS hostname and normal certificate/hostname validation. It does not change
system DNS, hosts files, certificates, remote URLs, split assignments or range/
CRC/SHA checks. Tests verify the original TLS hostname and IPv4 socket timeout
arguments. No global Python environment or active process is patched.

Use `python -m scripts.run_starss23_ipv4` with the same acquisition arguments
and a **new output directory**, after the original process has terminated or
the user has approved stopping it. First run without `--execute` to confirm
disk budget. The network workaround does not itself establish capacity or
complete the media acquisition. At this record's creation, permission to stop
the pending preflight had been requested and no replacement acquisition was
started.

## Downstream integrity gate

### Approved switch outcome

On 2026-10-11 the user approved stopping the old preflight and switching to
IPv4. At execution time PID 65896 was already absent: the original process
had naturally completed with exit code 0 and PREFLIGHT_PASS, taking 764.872 s.
No process was killed. The fresh IPv4 preflight in
`C:\CodexArtifacts\pretrained-reuse-r2\starss23-media-preflight-ipv4-01`
also passed, taking 7.364 s. Both read 38,446 ZIP-directory bytes and computed
identical acquisition sizes: 78 recordings, two reused recordings,
4,710,820,298 new uncompressed bytes and 3,881,505,469 compressed bytes.
Available disk was approximately 10.87 GB, sufficient for the required
2 GiB reserve. This is one observed transport comparison, not a general network
benchmark.

The actual IPv4 acquisition was then launched with `--execute` and new output
directory `C:\CodexArtifacts\pretrained-reuse-r2\starss23-frozen-media-ipv4-01`.
This launch is not an acquisition-completion claim. Its `progress.json` and
process exit status are authoritative; the downstream gate below must still
pass after all 78 pairs are present. The original preflight and smoke files
remain preserved. No learning job or Teacher artifact was touched.

Before feature extraction from a completed acquisition, run:

```powershell
python -m scripts.verify_starss23_acquisition --progress <completed-output>/progress.json --splits docs/pretrained_reuse/starss23-splits-v1 --summary docs/pretrained_reuse/R2_STARSS23_SPLITS_20261011.json
```

This rejects INDEXING/FETCHING/FAIL/PREFLIGHT_PASS outputs, changed summary
hashes, incomplete recording sets, wrong room/split/source identity, missing
audio/video, byte-size or SHA mismatches and identical underlying media reused
under different recording IDs. It checks the exact frozen set, not only a
reported count. Eight CPU tests cover successful paired input and rejection
paths. This guard has not certified the real 78-recording dataset yet because
the acquisition is still incomplete.

## Fixed-window manifest after acquisition

`scripts/build_starss23_window_manifest.py` first verifies the complete paired
acquisition and pinned metadata, then emits every complete, nonoverlapping
five-second interval from each recording. It uses the shorter audio/video
duration and records the discarded incomplete tail. It checks source audio
format and video timeline origin, preserves all twelve binary queries and the
frozen recording/room/split identities, and hashes each output JSONL. It stores
media references rather than full-resolution decoded arrays. Official dev-test
is excluded. Calibration/validation annotations are read only when this actual
window-generation step is run; their class counts are not used to redesign
the split or task.

The real in-progress acquisition was passed to this entry point as a negative
integration check: it raised `acquisition is incomplete` before creating an
output directory. No full-corpus window-generation success is claimed yet.
Five additional tests cover interval boundaries, shorter-media handling and
invalid durations. The full CPU suite passed 287 tests with one skipped;
Ruff passed.


## First full acquisition interruption and recovery

The IPv4 acquisition `starss23-frozen-media-ipv4-01` exited with code 1
at 14/78 completed pairs: `ValueError: range length mismatch` while fetching
`fold3_room13_mix005` video. The exact-range guard rejected a short body;
no incomplete video was accepted. The 14 completed pairs (28 files) were
rehashed and size-checked successfully in
`C:/CodexArtifacts/pretrained-reuse-r2/starss23-failed-acquisition-audit-01/result.json`.
The failed progress SHA-256 is
`c06721a9c0e7a7f086f4dd3c34769bd0d493a27253544ac17789be0fa7771fe1`.
Original files and the failed report remain unchanged. The unpaired saved audio
is retained but not reused by the completed-pair recovery path.

The acquisition CLI now accepts `--reuse-progress` from a terminal PASS/FAIL
acquisition with the exact frozen summary hash. It verifies complete-pair split
identity, file size/SHA, and duplicate media before reuse. This does not make a
partial acquisition eligible for training: the downstream all-78 gate is unchanged.
Only short bodies with correct HTTP 206 and Content-Range can be retried, twice
at most for the new acquisition; incorrect headers and oversized responses fail.
Every returned body, including failed short bodies, counts against the existing
transfer budget. TLS and ZIP CRC checks remain enabled. No generic network-error
retry or training resume is claimed.
