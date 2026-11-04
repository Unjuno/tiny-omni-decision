# Actual public AV media acquired and audited

Extends [fixed layout](R2_LOMBARD_GRID_LAYOUT_20261013.md); no new model-quality
evaluation or learning was run. Original layouts, ZIPs and failed attempts remain
unchanged. No official final/sealed audit was accessed; cloud cost0.

## Acquired source and real decode

Two fixed-ETag full ZIPs completed in784.90s using two HTTPS readers,900s cap,
3GiB disk reserve and exact Content-Length/ETag checks. All5,390 member CRCs
per archive passed and indices matched the previously pinned directories.

- Audio652,614,041bytes,SHA
  `e36707602fdbeaa2e2284acb29bba883261d5e86857f694ac40cbb56cdcc5e10`.
- Front837,239,327bytes,SHA
  `8c7ff24dffd5acb3b8cd1dceb4bd33463b82b797f66a8ed092ad681a9ffb6313`.

All5,390 audio records decode to finite/nonzero16kHz mono waveform; all are
within the30s frozen speech frontend limit.5,386 paired recordings fully decode
329,556 video frames with monotonic timestamps. Exact audio-file, video-file and
normalized mono-float waveform hashes each have5,390 unique values and zero
cross-split duplicates. This does not prove absence of edited excerpts or unknown
pretraining reuse. Last video PTS minus audio duration ranges[-.080917,+.165]s;
1/50/99percentiles[-.042125,0,+.080662]s. Exact AV offset is still unproven.
Do not adjust offsets from validation or manufacture noise to create joint benefit.

## Failures and resolution before any model scoring

First full decode terminated without a saved final report; its exact termination
cause remains unknown. Isolated16-record one/four-worker probes both passed,
weakening a blanket concurrency explanation. Full attempt02 added faulthandler
and source-ID logs and ended on `av.error.InvalidDataError` after79.22s.
It retained5,386 completed rows; four missing source IDs were identified:

`s32_l_pwip9p`, `s32_p_bwwj2n`, `s33_l_pwajza`, `s33_p_sgwq2s`.

All four are **training** records. Their MOVs lack`moov`/`avcC` metadata.
Fetched the official talker32/33 alternate ZIP members using exact ranges;
all four copies are byte-identical and equally undecodable. No inferred container
repair, substituted camera view, rewritten timestamp or changed label is used.

Create new `lombard-grid-layout-02-media-gated` before model scoring:

- Shared usable train4,080, identical IDs for all future audio/video/joint arms.
- Calibration398 and validation889 preserve original JSON **bytes**, IDs, labels
  and source/group assignments. No held-out example was removed.
- Original4,084 train layout remains intact;4 invalid training recordings retain
  their labels/provenance in`invalid-training-media.json` and the source ZIPs.
-19 unresolved labels also remain separately recorded. Common labeled+usable
  AV coverage5,367/5,390; this is coverage, not model Accuracy.

The media gate rejects incomplete/duplicate/unknown audit identities and refuses
to shrink either held-out partition on media failure. Train exclusions come from
physical media validity before metrics, not example difficulty or model behavior.
Tests cover these safeguards. Native slot vocabularies/option counts and speaker/
complete-phrase grouping are unchanged.

## Preview/full-release distinction and encoder audit

The four official preview audio files have fewer samples than same-ID archive
members. The initial archive/preview byte-identity assumption failed. Keep their
hashes/versions distinct: do not silently swap previews into the training corpus.
Independent frontend-reload attempt02 tested each version separately. All four
preview token outputs reproduce prior encoder smokes bitwise; all four full-release
inputs yield bitwise-equal old/reloaded frontend tensors and finite native tokens.
Parameters remain frozen FP32. The failed attempt01 and evidence are preserved.
No ASR/candidate-score parity across different waveforms is claimed.

Four full-release training utterances also passed a fresh CUDA feature smoke:
same-GPU replay bitwise exact, CPU maximum absolute difference5.51e-5 within
the predefined1e-4 tolerance. Frozen8,208,384-parameter Encoder,FP32,TF32 off,
seed17; allocator peaks100,234,752bytes allocated/113,246,208reserved.
No Decision learning or quality metrics were run. Saved under the Encoder's
`cuda-full-release-smoke-01/`; driver/context memory is excluded from these peaks.

## Saved artifacts and next execution

Under`C:/CodexArtifacts/pretrained-reuse-r2/`:
`lombard-grid-archives-01/`, `lombard-grid-full-decode-controller-02/`,
`lombard-grid-decoded-audit-02/`, `lombard-grid-broken-video-alternate-01/`,
`lombard-grid-media-integrity-01/`, `lombard-grid-layout-02-media-gated/`,
`lombard-grid-publisher-sample-parity-01/`. Adjacent JSON records all hashes,
commands/source runners, source versions, failures and new layout hashes.
Frontend audit:`C:/CodexArtifacts/pretrained-reuse-r1/whisper-encoder-runtime-01/frontend-reload-audit-02/`.

Next: measured frozen-feature extraction and fixed unimodal/joint native command
readouts. Use raw waveform/video only; no filename/transcript/alignment features.
This command-slot task adds a scoped AV decision test. It does not replace the
general image/audio/text/temporal-video battery, prove joint benefit or meet the
95–98% product target. The final goal remains active.
