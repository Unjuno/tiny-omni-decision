# AMI official media-host connectivity recheck (2026-10-22)

## Scope and outcome

Retried only the directory/header request for the next AMI meeting-level
candidate cohort from the local Windows host. The host resolved to
`129.215.202.26`, but both the ordinary HTTPS HEAD and an HTTPS HEAD pinned to
that IPv4 address timed out before response headers. The direct IPv4 attempt
did not establish TCP/TLS. No media, labels, parquet rows, or annotations were
downloaded or read; no GPU, training, or evaluation was used.

This confirms that the current local route cannot reach the official host. It
does not establish that the candidate assets are missing upstream. The new
attempt was written to a separate artifact directory and did not alter earlier
AMI attempts:

- Artifact: `C:\CodexArtifacts\pretrained-reuse-r2\ami-official-index-connectivity-attempt-10\connectivity.json`
- SHA-256: `5bb2e86fa03de85e5e9ec3fedd864e592d12a2cfe0dd469cecf0f990e89cd42c`
- Observed: `2026-10-21T21:13:56Z` (`2026-10-22 06:13:56 JST`)

## Exact bounded checks

1. PowerShell `Invoke-WebRequest -Method Head` to the official `TS3009a/video/`
   index timed out after the configured 15 seconds with no response headers.
2. `curl.exe --head --location --connect-timeout 8 --max-time 12 --resolve
   groups.inf.ed.ac.uk:443:129.215.202.26` to the same HTTPS index exited 28
   after the 8-second connection timeout. TLS was not established.

The checked path was an index only; no payload request was made. The source's
license reference remains the official AMI release terms, which state CC BY
4.0. Rights status alone does not admit a corpus: the six-meeting candidate
still needs verified media availability, four-camera coverage, meeting-specific
speaker/camera mapping, synchronized timestamps, and target visibility before
any quality fit. See the [AMI audiovisual preflight](R2_AMI_AV_BINDING_PREFLIGHT_20261022.md)
for the existing split and those gates.

## Decision

Keep the new AMI cohort **REVIEW / not admitted**. Do not start a model fit or
change the split on the basis of this connectivity attempt. Revisit media
acquisition only after the local route changes or a rights-compatible official
distribution route is independently verified. The current obstacle is network
reachability, not a negative model-quality result.

Sources: [official AMI download and release page](https://groups.inf.ed.ac.uk/ami/download/),
[official TS3009a video index](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/TS3009a/video/).
