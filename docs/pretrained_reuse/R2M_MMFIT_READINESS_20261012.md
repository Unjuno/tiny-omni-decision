# MM-Fit paired-source audit

The official [project](https://mmfit.github.io/) links sensor ZIP and separately
published RGB video. [Pinned video record7672767](https://zenodo.org/records/7672767)
declares CC-BY4.0 in the actual API response. It supplies21 RGB files with checksums;
no videos were downloaded. An earlier record also includes depth, so the latest
RGB-only revision must not silently be described as containing depth.

The1,742,309,258-byte sensor ZIP was inspected with bounded HTTP ranges, validating
206 status, exact Content-Range/length and If-Match ETag. Central-directory entries
were saved; no TXT/MD documentation members were present. No sensor arrays,
labels or media contents were loaded. Server version ID:
`siCk1n3EvPszsjcJsPsywP8ovDdgCW5U`; ETag
`c72b4cb675f8947c448aa57622f86af3-102` is a multipart identity, not a file SHA-256.

The starter-code repository at e4b6b7e1fff68e3d93130ed087d23593c9016c94 has an MIT
software license; that does not independently establish sensor-data rights.
Sensor source therefore remains REVIEW until an explicit applicable grant is
located. Do not transfer the video record's license to the separately hosted ZIP
without evidence. A synchronized recording alone also does not establish that
the proposed decision requires both modalities; matched unimodal controls remain
mandatory once a task is admitted.

Metadata and inventory are preserved under
`C:/CodexArtifacts/pretrained-reuse-r2/mmfit-source-audit-01`.
The bounded audit succeeded and avoided a1.7GB download. Real paired training
has not started; no quality, alignment accuracy or model capability is claimed.
