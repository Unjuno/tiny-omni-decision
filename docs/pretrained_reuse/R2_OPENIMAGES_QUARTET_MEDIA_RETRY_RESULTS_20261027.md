# Quartet media retry: rate limited

The [fixed retry protocol](R2_OPENIMAGES_QUARTET_MEDIA_RETRY_PROTOCOL_20261027.md)
waited more than five minutes after the prior acquisition completed, then
requested the recorded original URL for the photo whose thumbnail had returned
HTTP 410. The original URL matched the checked Flickr photo identity, but the
request returned **HTTP 429**. In accordance with the fixed stop condition,
the process stopped after this one request; it did not retry the other image
that had previously returned 429. No image was saved, no MD5 could be verified,
and no requests should be repeated while the provider's rate limit persists.

The earlier 12-group set still has ten complete quartets and two incomplete
ones. This is a media-acquisition limitation, not a model result. No candidate
was substituted, and the prior 46 JPEGs and all existing artifacts remain
unchanged. No training or evaluation occurred in this retry.

Private output:
`C:/CodexArtifacts/pretrained-reuse-r2/openimages-quartet-media-retry-20261027-attempt-01/`.
Raw SHA-256 of `summary.json`:
`914e8146b11facd1aacbe80e7572cc83321d280020e72f95679858c4654d8ff7`;
`private-retry-manifest.jsonl`:
`780dd77ead777951a000b2b89d0df642ea8bd3cbb7ae94fc596704b641df13a5`;
`private-duplicate-warnings.json`:
`a5338d955b09046ec0b16f3a9625b7955c763aae07dc722e474e6078745f932f`.
The output includes source, script and protocol hashes. The provider's
Retry-After header was not captured by the legacy download helper, so the
time at which it is safe to retry is unknown.
