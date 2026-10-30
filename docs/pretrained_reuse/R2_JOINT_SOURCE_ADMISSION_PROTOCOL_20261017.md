# Joint AV source admission: bounded physical probe

Continue the conditional-interaction work; its current common corpus has no joint
AV observations. New task supervision is required before gate training. Preserve
all existing runs, all earlier source rejections and the200M product objective.

## Current source evidence and selected next probe

- Existing Lombard Grid contains real pairs and known spoken slots, but the closed
  native-command joint models could ignore video. Repeating those fits is not the
  next experiment. STARSS23 is admitted but its event-presence task does not itself
  prove complementary visual necessity; existing room/positive-support gaps remain.
- Action Verb Corpus has a CC-BY-4.0 institutional page and Zenodo license field,
  but the same Zenodo record's version text says CC-BY-NC-SA4.0. Keep REVIEW; no
  media/annotation ingestion for training. Do not silently choose the permissive
  statement. Raw public metadata snapshots have been saved.
- FAIR-Play publisher says CC-BY4.0, roughly aligned pairs and random scene-overlap
  splits. Record it as a candidate; do not acquire100GB or invent semantic labels.
- Greatest Hits official author page links CC-BY4.0 and a21,142,601,241-byte ZIP.
  Its remote directory contains977 parent videos, each with six members (three
  MP4,two WAV,one times file), plus train/test ID lists. Exact names/CRC/offsets,
  strong ETag and734,903directory bytes have been saved. No member payload yet.

## Fixed physical probe, before labels/model outcomes

Archive: `https://web.eecs.umich.edu/~ahowens/vis/vis-data-256.zip`.
Size21,142,601,241; strong ETag `"4ec327e19-5cc7fa3687c80"`.
Directory evidence:`C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-directory-01`.

1. Read official train/test ID lists as split metadata only. Reject overlap,
   duplicates, absent parent members and unsafe paths. Do not read test media or
   test label payloads. Do not create a development split by random windows.
2. Sort TRAIN IDs by SHA256(`17:` + ID), choose the first three without looking at
   labels or quality. Fetch their times files. Keep all labels/unknown fields;
   inspect schema before defining a Decision target. Do not discard hard records.
3. Fetch the first selected TRAIN parent's existing `_denoised.mp4` only if it
   contains the actual released AV streams; decode every stream to inspect finite
   samples, monotonic PTS and duration. If no audio, record that and require the
   explicit corresponding audio member; do not fake synchronization. Inspect
   media names before acquisition to avoid an annotation-overlay video.
4. Exact HTTPS IPv4 ranges, If-Match,206/Content-Range/ETag validation, per-request
  25s wall cap and30s subprocess bound, total64MiB transfer cap, member32MiB
  uncompressed cap. CRC and SHA256 every extracted member. Refuse existing output,
   changed source version, malformed ZIP header or decompression overrun. No full
   archive hash claim from partial ranges; preserve failed attempts.
5. Save executed source/config/commands, source and per-file hashes, actual schema,
   decoded timings and explicit admission limits. No model, optimizer, quality
   scoring, calibration or sealed audit. CPU only and no dependency installation.

This probe must establish what the released label bytes actually contain. The
paper's annotations or licensing alone cannot certify a trainable joint benchmark.
Material/action recognition is not automatically a both-required task. The next
decision depends on measured schema, groupability and usable complementary targets.

## Tests and publication

Test exact-range/version rejection, bounded decompression, CRC/name checks and
deterministic train-only selection before acquisition code runs. Record the real
probe and source decision, then Ruff/CPU tests/manifests, commit/push and exact CI.
Treat schema or source failures as evidence, never as a95–98% product conclusion.
