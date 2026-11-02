# iNaturalist CC0 cohort gate recheck (2026-11-02)

## Disposition

The two existing small iNaturalist bird metadata cohorts do not meet the
project's provisional minimum for a quality PASS and are not admitted for
another image readout. No photos were downloaded or decoded, no model was
loaded or trained, no metrics were computed, and no sealed audit was accessed.
This rejects these frozen metadata slices, not the entire iNaturalist source.

The applicable policy requires at least 1,000 evaluation examples and 200
independent source groups before a task can receive a PASS decision. The
current metadata slices are far below both thresholds even after granting the
stored grouping keys the strongest plausible interpretation.

## Read-only metadata evidence

The connected metadata candidate contains 1,570 rows with SHA-256
`99311045cdc99ce6cf3596149983c595706e3b7dcf018c21337e59240f3a1e9e`.
Filtering strictly to records where both observation and selected-photo
licenses are `cc0` leaves 83 training and 7 validation images. Those rows span
8 and 3 connected-component keys, respectively; validation covers only 3 of
the 8 classes. The parent summary marks this cohort
`CONNECTED_METADATA_NOT_ADMITTED` and records 26 source-row join failures.

The separate 1,596-row development snapshot has SHA-256
`1d97f5368855c80abdd98bfb95cdf0c6359b21bef7c8ab580318f3b3c914801d`.
The same strict two-license filter leaves 64 training and 26 validation
images, spanning 7 and 4 stored observer keys. The snapshot is explicitly
`DEVELOPMENT_METADATA_NOT_ADMITTED`; its parent record says the live API
snapshot is nonrandom and photo rights, image identity and deduplication had
not been verified.

The existing metadata files also do not directly satisfy the proposed
per-record media-download manifest contract: they lack the required hashed
observer-group field and are not the frozen `train.jsonl` / `validation.jsonl`
pair. A downloader must not infer or silently fill those fields. The recorded
pages 41–50 expansion remains rejected as well: it yielded only 27 training
and 14 validation records across two classes after identity exclusions, below
the predeclared cohort floor.

## Next gate

Do not run the image downloader, feature extractor or readout on these
metadata slices. A later iNaturalist attempt would need a newly frozen cohort
with per-record CC0 observation and photo licenses, verified photo/observer
identity, zero cross-split identity overlap, at least 1,000 evaluation images
and 200 independent validation groups, and all required classes represented.
If a source-level split cannot supply that coverage, choose another
rights-cleared image task rather than relaxing the gate. No sealed-audit IDs or
media should be used for cohort repair.
