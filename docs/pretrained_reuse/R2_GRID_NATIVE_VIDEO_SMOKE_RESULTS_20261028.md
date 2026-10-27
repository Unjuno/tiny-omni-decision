# GRID original-video native encoder smoke

The [fixed 16-example technical smoke](R2_GRID_INPUT_SMOKE_PROTOCOL_20261028.md)
used the original `s1.zip`. The 423,520,514-byte archive matched publisher MD5
`cbd6556668f061b5c3681bc722659b39` and local SHA-256
`2eec254832a87bb7158e8d26ec0f8f1020b79739d09ef3ae325d294b51ca41f8`.
Despite the Zenodo description's reference to JPEG videos, the selected files
are MPEG-1 videos within the ZIP. All 16 member CRCs and individual SHA-256s
were checked; all decoded to 75 ordered frames at 360×288, 25 fps, with 2.96 s
between the first and last frame timestamps. The verification receipt is
`C:/CodexArtifacts/pretrained-reuse-r2/grid-s1-video-smoke-20261028-attempt-01.json`,
SHA-256 `5317e0841fbee8645298c7dc4c0466f5baf3b0925a15e840866c113d94f9f8d1`.

The existing pinned native VideoPrism worker encoded all 16 on CPU. It
returned finite `16×768` pooled temporal features for each; all 16 feature
hashes differ. The worker completed its dependency and before/after weight
checks with `PASS_NATIVE_VIDEO_WORKER`; its full checkpoint has 114,365,184
parameters. The controller took 181.62 seconds including initialization and
terminal checks. No GPU, training, audio, final Decision scoring, or held-out
quality evaluation was performed. The data comprises one speaker and a narrow
lexicographic sample; feature uniqueness is not evidence of task performance.

Output and media stay outside Git:
`C:/CodexArtifacts/pretrained-reuse-r2/grid-s1-native-video-20261028-attempt-01/`.
`summary.json` SHA-256
`41f1af1e0213179d6cec7249f02532a2aa34705feb50836a4bfc74ee362b73d9`;
the worker result, per-video features, raw selected files and log are retained
there. The separate audio archive was still downloading when this result was
recorded. Two timestamp tests, one feature-shape/finiteness test and Ruff
passed. Exact synchronized audio/video behavior and the correspondence task
remain unverified.

## Frozen video decision head

The existing Lombard `video_ordered` head and normalizer were loaded by their
previously recorded hashes and applied to all 16 verified GRID temporal
features on CPU. All outputs had the expected finite 51 logits; the head
contains 106,355 registered parameters. No labels were opened or scored, no
audio was supplied and no weight was changed. This verifies a component
interface only; it does not establish accurate choices or audiovisual fusion.

The new output is
`C:/CodexArtifacts/pretrained-reuse-r2/grid-s1-video-head-20261028-attempt-01/`;
`summary.json` SHA-256
`b063f9462f80d91e5d7957f85c256cf98f5835f38f48ddf647303ea64d73b093`,
and `video_logits.npy` SHA-256
`7cc918e3c36e39fd29da10022ae4b0e1174e22b1e5f729a9d2cd3532afd39bea`.
Input feature hashes and head/normalizer/source hashes were rechecked. The
finite-logit CPU test and Ruff passed.
