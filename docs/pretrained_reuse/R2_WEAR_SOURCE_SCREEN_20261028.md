# WEAR synchronized video/IMU source screen (2026-10-28)

Status: **not admitted; no download and no fit**.

## Task fit

WEAR is well matched to a narrow synchronized video-plus-accelerometer
decision task: 22 participants performed 18 workout activities at 11 outdoor
locations, with four 50 Hz watches and head-mounted egocentric video. Sessions
are uninterrupted and activity order is participant-selected, so a
prospective temporal decision benchmark is plausible. The official page
separates 14 MB annotations, 44 GB processed features, and 164 GB raw data.

## Exclusion

The official dataset page licenses WEAR under CC BY-NC-SA 4.0 and expressly
prohibits commercial use. This conflicts with the project's product-directed
training objective. The raw corpus is also 164 GB, exceeding present local
storage. Do not use its data, derived features, or released annotations for
training or model selection. No files were downloaded.

## Primary source

- [Official WEAR project and dataset page](https://mariusbock.github.io/wear/)
  (dataset description, payload sizes, and license)

## Decision

Reject this source for the current product path. Reconsider only if the rights
holder supplies compatible written terms and the project has adequate local
storage; a commercial-use exception cannot be inferred from the paper's
research results or code license.
