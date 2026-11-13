# Human-human handover video/IMU source and task screen — 2026-10-31

## Disposition

`REVIEW_NOT_ADMITTED` for the R2-M joint-quality experiment. This source is a
promising small, synchronized video-plus-sensor candidate, but its published
labels and grouped evaluation are not yet sufficient to define a task that
requires both inputs. No files, annotations, questionnaires, or media were
downloaded or opened, and no model was trained or evaluated.

## Pinned public records inspected

- Mendeley Data version 3, DOI `10.17632/cyr9gmhspy.3`, published 2020-04-20:
  <https://data.mendeley.com/datasets/cyr9gmhspy/3>
- Associated Data in Brief article, DOI `10.1016/j.dib.2018.11.110`:
  <https://www.sciencedirect.com/science/article/pii/S2352340918315038>

## Metadata-only file inventory follow-up

The official Mendeley Data public snapshot API confirms version 3,
`is_metadata_only=false`, and CC BY 4.0. The public root-folder file listing
returned eight completed entries: seven `.m` files and one `.txt` file, with a
combined listed size of 20,631 bytes. No video, sensor spreadsheet, or
questionnaire file appears in that root listing. This does **not** prove the
recordings are absent: the file-list endpoint queried only `folder_id=root`,
and the official folders endpoint requires OAuth. An unauthenticated read-only
request to that folders endpoint returned HTTP 401, so nested folders and
their contents remain unverified. The public Mendeley page also rendered an
empty file list in the available text view. No file-content or zip-download
endpoint was called.

Three metadata-list request timings were 2.484 s, 1.123 s, and 1.113 s
(mean 1.573 s). This estimates only repeated metadata-list requests; it is not
a media-download estimate and does not justify downloading the dataset.

The version 3 dataset page explicitly labels the dataset files CC BY 4.0. The
associated article's full text is marked CC BY-NC-ND; that article license is
not treated as the dataset license. Use would require attributing the dataset
and keeping the article and its figures outside any derivative/data use unless
separately permitted.

## Task-relevant facts from the publisher and data article

- The release describes 1,087 recordings from 18 volunteers across 76 test
  configurations. A configuration varies role, starting position, object, and
  in some conditions the experimenter's strategy.
- Each recording has timestamped synchronized sensor measurements and an
  associated video. The paper describes a 47-dimensional sensor representation
  (motion-capture markers, one person's Kinect skeleton, and two wrist IMUs),
  approximately 6.6–7 Hz sensor acquisition, and 8 fps video.
- Single-blind recordings have six participant ratings: naturalness,
  practicality, comfort, safety, speed, and timing. Each is a four-point
  Likert response. The authors report that volunteers interpreted the rating
  scale differently, so these are not automatically consistent cross-person
  ground-truth labels.
- Double-blind recordings include participant rankings of object
  manipulability. The reviewed material does not establish a fixed supplied-
  option Decision task or show that both video and inertial signals are
  necessary to answer it.
- The data article says folder names encode a volunteer for single-blind data
  and a volunteer pair for double-blind data. Exact participant-pair connected
  components, file-level group counts, session overlaps, and a safe independent
  train/validation/audit split have not been reconstructed.

## Why it is not admitted yet

The source could support a narrow human-handover perception study, but fitting
on subjective ratings risks learning rater-specific scale interpretation,
configuration identity, or participant-pair shortcuts. A random recording
split would leak shared people and repeated configurations. It is also unknown
whether a participant-disjoint split leaves enough independent groups for
validation, and whether the released videos/sensor files satisfy the repository's
privacy/consent and artifact-retention policy at the needed granularity.

Before any payload access, require a metadata-only file inventory with sizes and
checksums, reconstruct participant/pair connected components, establish a
predeclared target with stable label meaning, and demonstrate using metadata
controls that the task cannot be solved from condition IDs or option priors.
Only then consider a bounded train/validation task and matched video-only,
sensor-only, shuffled, missing, and stale-input controls. Keep any test/audit
partition unopened until architecture and selection are frozen.

## Source references

- [Mendeley Data v3 record](https://data.mendeley.com/datasets/cyr9gmhspy/3)
- [Mendeley Data API documentation](https://data.mendeley.com/api/docs/)
- [Data in Brief article](https://www.sciencedirect.com/science/article/pii/S2352340918315038)
