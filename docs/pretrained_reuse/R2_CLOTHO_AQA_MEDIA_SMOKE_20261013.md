# Clotho-AQA media smoke and full index coverage

Executed a bounded HTTPS range acquisition against the official record6473207
audio_files.zip (3,140,677,494 bytes). Selected one training source by minimum
SHA-256(`17:<source_id>`), independent of questions/answers. Kept TLS validation,
exact206 range checks,32MiB transfer budget and16MiB member-size bound.

Source155281, uploaderbmoreno, CC-BY3.0, member
`audio_files/0232 Street_birds_2.wav` decoded successfully: mono44.1kHz,
1,132,430 samples,25.678685seconds, finite and nonzero. ZIP CRC passed;
memberSHA256 `28ed6d57e46fe3da60e712bbdb300f3e8fa96b92f1960a6a05f6b13b4d639aa6`.
Transferred2,073,181bytes including directory reads. Partial acquisition does
not verify the whole archive's MD5. No model was loaded or trained.

Archive-index matching for all frozen development sources found855/856 exact
basenames after HTML entity decoding. Calibration138/138 and validation140/140
match; train577/578 match. The unresolved train filename is
`souffle_me_tallique.wav`; the index contains `souffle_me.tallique.wav`.
Do not silently skip the question or rewrite frozen manifests. An explicit
provenance-backed alias resolution is required before full acquisition.

The855 matched files total1,349,610,516 compressed bytes and1,697,766,684 decoded
container bytes (not feature-memory requirements). Free C: space at preflight
was15,706,230,784bytes. Full acquisition is not started. Directory names for
other splits were listed; only the selected training WAV content was read.
Official validation/test question files and sealed audit remain unopened.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-media-smoke-01/`
contains audio.wav, result.json, index.json and coverage.json. Exact executed
runner is saved locally as `clotho_aqa_media_smoke.py` in the parent directory.
The compact result and coverage accompany this report. Next: resolve the single
filename discrepancy using official source metadata, then bounded acquisition
and waveform integrity checks before model comparisons.
