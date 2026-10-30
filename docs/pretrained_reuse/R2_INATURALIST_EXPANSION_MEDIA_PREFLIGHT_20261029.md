# iNaturalist expansion media preflight (2026-10-29)

## Bounded sample

The combined pages 11–30 frozen expansion contains 921 train and 446
validation records (1,367 total). Before full acquisition, three deterministic
train rows were passed through the existing HTTPS-only, host/redirect-checked,
8 MiB-bounded downloader in a separate output directory. The downloader
verified MIME type and decoding, stripped EXIF in normalized output, and saved
raw-download, normalized RGB, media and perceptual hashes. No image was viewed,
no model was run, and no feature or quality metric was produced.

All three pilot records happen to be Rock Pigeon because the deterministic
selection takes the first rows in sorted manifest order. This is a throughput
and pipeline smoke, not a taxon-representative storage sample. For a broader
storage estimate, use the prior complete 176-image cohort's observed normalized
PNG mean as well.

| Pilot image | Downloaded bytes | Normalized PNG bytes | Processing seconds |
|---|---:|---:|---:|
| photo `170944081` | 188,310 | 413,418 | 1.469 |
| photo `172018955` | 186,370 | 401,423 | 1.484 |
| photo `175429656` | 104,740 | 318,146 | 1.250 |
| **Mean** | **159,807** | **377,662** | **1.401** |

Total wall time was 6.313 seconds, including two deliberate 1.05-second
request-spacing waits. These three samples alone project 0.481 GiB normalized
storage for 1,367 rows. The earlier full 176-image cohort averaged 340,230
normalized bytes/photo and took 242.0 seconds end-to-end; applying its mean
size gives 0.433 GiB, while its observed throughput projects about 31.3 minutes
for 1,367 rows. Use **0.49 GiB storage** and **20–35 minutes acquisition time**
as planning estimates, not guarantees. The downloader retains normalized PNGs;
temporary raw responses are removed after decoding.

## Current resource gate

At the post-pilot check, C: had 0.263 GiB free and no other local filesystem
volume was available. Keeping the established 2 GiB safety reserve, the
estimated normalized image payload, and modest temporary/metadata overhead
requires about 2.6 GiB free before full acquisition. Do not start the 1,367-row
download or model fit under the current space condition. The Orbitune process
was still active on the RTX 3080 Laptop; no Tiny Omni GPU job was started or
existing process touched.

## Reproducibility

- Combined input manifest SHA-256: `362a562568910676a15b71d42233fbeb8f6811116afcb6ef22f846bcf83e8618`
- Pilot directory: `C:\CodexArtifacts\inat-birds-cc0-expansion-media-pilot-20261029-attempt-01\`
- Pilot receipt SHA-256: `e6f6f01c89f2df584a5b6aaaf06d92fa6ca55b1e550c296323e37d69bc5f7be3`
- Source data remains external to Git. The original media cache, features,
  trained readout and legacy validation remain unchanged.
