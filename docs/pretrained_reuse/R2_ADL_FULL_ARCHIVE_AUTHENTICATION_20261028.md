# R2 ADL full archive authentication

## Result

On 2026-10-28, all 25 activity ZIPs in Zenodo record `10.5281/zenodo.7937591`
were present locally and independently rehashed. Each exact byte length and
publisher MD5 matched the pinned record; each SHA-256 matched the acquisition
receipt. The corpus totals **4,572,316,824 bytes**. This clears archive-level
integrity for these files only. It does not establish row-level integrity,
participant/session independence, physical synchronization, task suitability,
or product-quality evidence.

The 23 newly fetched archives total **4,433,512,911 bytes**. Two archives from
the prior attempts were reused only after exact-size and publisher-MD5 checks.
All archives remain outside Git under
`C:/CodexArtifacts/pretrained-reuse-r2/adl-full-archive-verification-20261028-attempt-01/`.
No media or model weights were added to the repository.

## Acquisition and verification

- Source record SHA-256: `c01bf60460981280c5063ea54c038d41c751bf3056220059c3896cbc5d2b97c6`
- Download plan SHA-256: `a387e837d9e14ea00fec20e72c07331fafd3f12e07661cf671da0f4216152dc2`
- Download receipt SHA-256: `af284fe02e447a33194aa0fa871a0601dc0f067ec81137ceb9a17843379230aa`
- Downloader SHA-256: `5db7d2d002c7ceaae8d806fa08a2f1b3803ef034fbbbc9a03adf1acf7946b3e4`
- Downloader-test SHA-256: `0dfde54404e55799e3970e2b8613fd2095effa12ee9f2735b6d99aefa4e2fb1`
- Result: **25/25** publisher MD5 checks and **25/25** local SHA-256 rechecks
- New downloads: **23**, **4,433,512,911 bytes**
- Sum of per-download measured elapsed times: **10,137.344 seconds**
- Effective throughput across those bytes and measured times: **0.4119 MiB/s**
- No cloud compute or GPU work was used.

The first three transfer samples measured 0.2346, 0.2662 and 0.1091 MiB/s,
for a byte-weighted mean of **0.2136 MiB/s**. The third sample resumed a
previously retained 55,574,528-byte partial file; it transferred the remaining
22,824,594 bytes and was included using transferred bytes and request-to-finish
elapsed time. That sample-based estimate projected 18,771.6 seconds for the
then-remaining 4,204,577,686 bytes. Actual full acquisition completed with
higher later throughput. A Zenodo read timeout occurred once; the partial was
preserved and resumed only after adding exact HTTP Range validation. The
resumed archive was fully checked against publisher MD5 and SHA-256.

The downloader refuses an unexpected source record, metadata, file size,
publisher MD5, or Range response. A partial is appended only when the server
returns the exact requested byte interval. It is promoted to a complete file
only after full-size and publisher-MD5 validation. Focused tests cover exact
Range continuation and preservation of a partial when the server ignores the
Range request.

## Scope and limits

This is a source-file acquisition and cryptographic integrity result, not an
ADL dataset admission decision. The existing source screen still reports only
limited activity-level participant/trial and sensor identity checks; the
complete 25-activity HDF5 content inventory, all-sensor row correspondence,
cross-activity participant/session analysis, waveform alignment, and
representative class support remain to be audited. In particular, the
previously observed sparse `14_sweep` support remains a limitation and must not
be repaired by duplicating examples or dropping the class.

No HDF5 members were extracted by this acquisition. No model was loaded, fit,
or evaluated; no validation or sealed-audit data was used. This does not add
video support: the ADL release contains audio and sensor channels, not released
video. Thus it cannot by itself satisfy the multimodal product objective.

