# VIDIMU video acquisition and frozen feature extraction

## Verified acquisition

The IPv4 download exited 0 after approximately 45.6 minutes. The complete
videosmallsize.zip has 1,060,471,682 bytes; publisher MD5
`b90c4ae792b3dd54a8ba337ae5f75e36` matched before opening the archive.
SHA256: `5224354912d0e72b0e6307c344ee6386d27bbd48fd26352d820a091c99fea9b7`.
Evidence: `C:/CodexArtifacts/pretrained-reuse-r2/vidimu-small-video-audit-01`.
The ZIP contains both original and bodytrack variants. Only original activity
videos are used; Npose calibration videos are not model inputs.

## One explicit filename alias

Initial exact matching found 129/130 train and 39/39 validation videos. The
missing `S41_A03_T01.mp4` corresponds by provenance evidence to the available
`S41_A03_P01.mp4`: the publisher's dataset.zip contains both P01 and T01
video-derived CSVs, and both the ordinary pair and Npose pair are byte-identical.
Ordinary CSV SHA256:
`f86bcd998ae7c9e43c6394a6d277152fbccb34bbe6cd5a808f0dfeeb0541dddb`.
Npose CSV SHA256:
`c12a76253548d5a6dc78accb6e49198325f67db66598003078ca3817f56cfb07`.
The upstream synchronization log uses T01. The recoding notebook at commit
649fdf07513fddcf5b0462a1bd888a5b13319c9c preserves input basenames.
This supports an explicitly recorded alias, not a pixel comparison against an
unavailable T01 video. Evidence: `vidimu-video-name-audit-01/csv-identity.json`.
No trial was dropped or moved between subjects/splits.

Mapping: `vidimu-video-mapping-02/mapping.json`, SHA256
`5b87e50675cc5234341dbc3ec7bc9febe6abfaa0130570e7a1bf1a4c4d8b5532`.
The earlier failed exact-match attempt is retained at `vidimu-video-mapping-01`.
All artifact directories here are below `C:/CodexArtifacts/pretrained-reuse-r2`.

## Actual extraction

A separate-process training-video smoke decoded 1,089 frames and selected eight
endpoint-spaced frames with actual presentation times from 0 to 36,266.67ms.
Frozen VideoPrism returned 1x2048x768 tokens; spatial pooling separately for each
of eight temporal positions produced 8x768 finite features. Initial compiled
forward took 6.225 seconds on CPU. No sensor clock alignment is inferred.

Full extraction command (PowerShell, repository root):

```powershell
$env:PYTHONPATH = 'C:/CodexWorktrees/pretrained-reuse-omni/src'
& 'C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv/Scripts/python.exe' -m scripts.cache_vidimu_video
```

An initial invocation without PYTHONPATH failed at import, before output creation;
no packages were upgraded. The corrected invocation exited 0. Source HEAD was
2141790d7e4bfbaa454b84b5c86c8e396528de80 plus the new extraction script, whose
executed SHA256 is `e9344886a372ab6a58b2d053fcc90672b82550e37bbba1b98744fb62e66f6ae6`.
Checkpoint and upstream source pins are checked by the script.

Output: `vidimu-video-features-01`, 130 train and 39 validation episodes.
Measured loop duration: 296.578 seconds; process peak working set: 4,156,506,112
bytes. Device: TFRT_CPU_0. Frozen encoder parameters: 114,365,184. This excludes
future readout parameters, which must also be counted in the resident total.
Training cache SHA256:
`a9d952fc39d1dfd413f604a033f66a6ef5dcb21f9583eeb25be9acf9be1d3c83`.
Validation cache SHA256:
`4a05d4e118482f980bbbcc9daf4037c5b461fb6e16a2c33072dff515af2f79b5`.
Independent post-run verification checked cache hashes, finite float features,
shapes 130x8x768 and 39x8x768, and exact mapping order. Cross-split exact video
byte-hash overlap is zero. This is not a near-duplicate-content guarantee.

The sealed audit payload was neither extracted nor decoded. No video/joint
readout training or quality comparison has run yet. The user requested time for
disk cleanup; additional training is held pending their completion message.
Existing sensor results, frozen corpus, legacy weights and splits are preserved.
