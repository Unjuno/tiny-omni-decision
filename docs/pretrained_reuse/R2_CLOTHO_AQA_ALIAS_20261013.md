# Original metadata check and bounded acquisition

The original Clotho release record4783391 development metadata was downloaded
and checked against its official MD5. SHA-256:
`b054a8d9d0f88436e7cf6341c82a70e9e975c3b3ddd560d9b04f2cd5fdc75949`.
The row for source16854 also uses `souffle_me_tallique.wav`, uploaderdeegnot,
CC0, matching AQA metadata. The AQA ZIP instead uses a dot. This corroborates
source metadata but does not prove that differently named ZIP member's identity.
No automatic fuzzy substitution or frozen-split modification was made.
Original metadata bytes contain non-UTF8 elsewhere; the relevant ASCII row was
inspected directly after strict UTF8 decoding failed, without replacing bytes.

Evidence is in
`C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-alias-audit-01/`.
Official source: https://zenodo.org/records/4783391 . Only development metadata
was retrieved; no original validation/evaluation metadata or audio was opened.

A separate bounded acquisition selects the first16 unique training source IDs
by SHA-256(`17:<source_id>`), and requires exact filename matches. It is a
pipeline probe, not a reduced training/evaluation corpus. The full frozen
856-source comparison remains unchanged and has not started training.
Runner: `python -m scripts.fetch_clotho_training_probe`;32/64MiB distinctions:
the earlier one-clip smoke used32MiB, this16-clip probe allows64MiB transferred.
Each member requires CRC, hash, allowed license and maximum16MiB; disk reserve2GiB.

Probe completed16/16. Transferred26,300,804bytes. Independently rechecked saved
file hashes and decoded every clip with PyAV: total362.098435seconds, finite
nonzero samples,16distinct decoded-array hashes. This is not a near-duplicate
waveform audit. Full acquisition result and decode report reside in
`C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-training-probe-01/`;
the compact decode report accompanies this document. No model training occurred.

CI follow-up found that the archived admission audit's Python formatting caused
Ruff failures. The repository copy was formatted, keeping the original local
executed source unchanged. Non-import AST equality was checked. Repository-wide
Ruff passed,764 CPU tests passed with1skip, and all25 CI manifest commands passed.
This repairs a reporting-code defect; it does not change any measured model result.

The user confirmed continuing public-data task selection. Future multimodal
claims still require matched unimodal controls and demonstrated benefit; audio
question answering alone cannot establish synchronized audio/video/sensor quality.
