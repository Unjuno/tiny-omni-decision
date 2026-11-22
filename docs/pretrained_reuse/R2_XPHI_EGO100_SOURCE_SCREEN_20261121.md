# xPhi-Ego-100 source screen (2026-11-21)

## Disposition

**`REVIEW_NOT_ADMITTED` for R2-M quality fitting.** This is a public-card-only
screen. No access form was submitted, no account details were provided, and no
metadata parquet, annotation, media, or sensor payload was opened or downloaded.

## Evidence and fit

The official Hugging Face card describes 2,012 clips (100.1 hours) from 152
workers at eight industrial/craft sites, with synchronized head-mounted video
and IMU. It declares Apache-2.0 and says participants gave recorded consent.
The whole release is listed as 195.5 GB. Access requires a logged-in request
form asking for real work email and phone; it is not anonymously downloadable.
The public card is mutable and this screen did not resolve an immutable
repository revision or file hashes.

The metadata includes participant, site, task and clip IDs. Participant-level
grouping appears possible in principle, but the 152 workers fall below the
project's provisional 200-group screen, and eight sites are too few for a
strong site-held-out claim. A split file and actual cross-task worker support
were not inspected. The eight trade labels are strongly tied to visible work
and environment, so video alone may solve a coarse task; the card does not
establish an action-choice task requiring both video and IMU. Action labels
are model-generated and are being added shard by shard, so coverage and label
quality are unknown. The release has no audio.

These constraints make it a poor next quality-bearing task for the current
goal. Do not submit personal details or access payloads unless the user
explicitly approves the account request and the project reopens this screen.
If reconsidered, first pin the repository revision, inspect only permitted
metadata, establish participant/site/task support and label provenance, then
freeze a group-disjoint task and storage budget before media acquisition.

## Primary source

- [Official xPhi-Ego-100 dataset card](https://huggingface.co/datasets/xphirobotics/xPhi-Ego-100)
