# CaptainCook training-only container probe

Follow-up to the release audit: lack of standalone audio links does **not**
establish lack of embedded audio. Selected recording `10_16`, lexicographically
first in the pinned official training-ID list, before opening any media. No
selection on quality, outcome or duration. No evaluation media/labels accessed.

Fetched only its explicitly listed GoPro 360p MP4 using requests with TLS
verification, 15s connection/30s read timeouts and a 150 MiB download cap.
Downloaded 104,842,046 bytes, SHA-256:
`c243764b1937aa5cdca29c758dde0cb8f4c7bf24bce84994d0229ef97657ce80`.
File and response headers are retained under
`C:/CodexArtifacts/pretrained-reuse-r2/captaincook-media-probe-01/`.

PyAV inspected three streams: H.264 video (time base 1/30000), AAC audio
(1/48000), and a data stream with no codec context. The first inspection failed
because it assumed every stream has a codec context. Retried inspection on the
same completed file with a null-context guard; no redownload or file replacement.

Decoded the first 48 audio frames: all finite, first PTS 0, last PTS 48128,
maximum frame RMS 0.024788. This establishes real decodable, nonzero audio for
this recording, not meaningful semantic content or dataset-wide coverage.
No listening, task-label reading, feature learning or model evaluation occurred.
No GPU or paid compute was used.

The companion JSON preserves all stream metadata and decoded-frame measurements.
Standalone Hololens microphone links remain absent in the pinned manifest.
Cross-device synchronization with IMU and the person-ID metadata disagreement
remain unresolved. This record has no synchronized accelerometer link, so do
not present this probe as validated audio/video/sensor fusion. Next gate is an
identity-provenance and clock-format audit before constructing a learning split.
