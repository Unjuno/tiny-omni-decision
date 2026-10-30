# EgoViz-120 source screen — 2026-10-29

## Decision

**Status: REVIEW; not admitted for training or quality claims.** EgoViz-120 is a
potential video-plus-sensor source, not an audio-video dataset. Its publisher
states CC BY 4.0 for the dataset and describes consented, face-blurred recordings.
The card describes synchronized cameras, IMUs and action annotations. The
available task labels are model-generated and not human-verified, the payload is
7.25 TB, and the hosted dataset repository itself has no files; the content is
served from a separate Hugging Face Bucket. These limitations prevent using it
as evidence for high-accuracy Decision quality now.

## Primary-source claims and metadata-only verification

- The publisher's [dataset card](https://huggingface.co/datasets/humaidtech/EgoViz-120)
  states 5,392 clips, 12 activity categories, 120.4 hours, 7.25 TB total, and
  that each MCAP contains synchronized streams. It identifies `group_id` as a
  capture session grouping the same worker, site and lighting, and `sequence_id`
  plus `index` as the original continuous sequence order.
- The publisher's [LICENSE file](https://github.com/humaidtech/EgoViz-120/blob/main/LICENSE)
  explicitly grants CC BY 4.0, including commercial use with attribution. The
  card also states that operators consented to public release and footage was
  automatically face-blurred. This is the publisher's license/consent claim;
  no media was opened to verify implementation.
- The [official storage bucket](https://huggingface.co/buckets/humaidtech/EgoViz-120)
  lists 7.25 TB in 16,180 files and an 871 KB Parquet index. The ordinary Hub
  dataset page says it has no uploaded files; media is therefore a separate
  bucket payload. No access token, paid service or media download was used.
- Downloaded only `index.parquet` (870,961 bytes) from the bucket on
  2026-10-29 02:16 JST. Its SHA-256 / HTTP ETag content digest is
  `c83126637cc24a7ac2b3600724b50cf9c53966b9b098acc57d5534e14ade1074`.
  The file and `metadata-audit.json` are preserved outside Git at
  `C:\CodexArtifacts\pretrained-reuse-r2\egoviz-source-screen-01\`.
- The Parquet index contains 5,392 unique `chunk_id`s and 180 `group_id`s;
  369 `(group_id, sequence_id)` pairs; no `sequence_id` appears under multiple
  groups. Group sizes range from 1 to 203 clips (median 25.5). All 12 categories
  occur in at least 10 groups; `cooking_food` occurs in 10, while
  `mopping_floor` accounts for 1,595 clips. These are metadata counts, not
  independent participant counts.
- The metadata states no official train/validation/test assignment. A future
  partition must keep every session group intact. The card does not establish
  that different `group_id`s correspond to different workers, so participant-
  independent generalization remains unverified. There are only 180 stated
  session groups, below the roadmap's provisional 200-independent-scene floor.

## Task and storage limits

The official stream list includes three cameras, depth, three IMUs, hand pose
and action topics; it lists no microphone/audio topic. The task could only inform
the optional video/sensor path. Activity labels are model-generated, not
human-verified, and the skill/object vocabularies are open. Since the frame
labels are also written to the MCAP action topic, future extraction must keep
the labels out of model inputs and use only raw camera and sensor topics.

The card's MP4 is a dashboard preview that overlays action labels, skeletons and
sensor visualizations; it is not a safe raw-video input for evaluating action
labels. A true video arm would need raw camera packets from an MCAP. The smallest
MCAP recorded in the index is 150,184,348 bytes and the median is 668,009,041
bytes; full MCAP payload is 6,805,356,013,034 bytes. At the acquisition check,
C: had only about 0.74 GiB free. No MCAP or preview video was downloaded.

Before any task experiment, the following remain necessary:

1. Inspect one bounded MCAP from a deterministic session to verify schema,
   camera/sensor timestamps, action-label timing, and that raw camera streams
   can be read without dashboard overlays.
2. Define video-only, sensor-only and joint Decision tasks with a question-only
   prior, group-disjoint split and interventions. The existing activity category
   alone does not show that both input streams are necessary.
3. Review generated label quality and estimate the minimum group-held-out sample
   needed. Do not call the 180 sessions 180 independent participants.
4. Recheck storage before downloading: even the smallest indexed MCAP is about
   150 MB, while the typical file is larger than the present free space permits
   for a safe extraction and replay.

This screen used metadata only; it trained no model, evaluated no quality set,
and did not access a sealed audit. It does not close R2-M or justify architecture
promotion.

