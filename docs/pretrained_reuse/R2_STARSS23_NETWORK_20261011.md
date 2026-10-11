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
