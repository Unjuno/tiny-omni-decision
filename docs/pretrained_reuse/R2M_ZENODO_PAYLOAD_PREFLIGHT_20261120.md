# R2-M audio + vibration payload preflight (2026-11-20)

## Acquisition and source inventory

The approved Zenodo record `10.5281/zenodo.7937591` was downloaded outside the
repository to
`D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1`.
All 25 publisher archives (4,572,316,824 bytes) passed the publisher MD5 and
local SHA-256 checks. The per-file receipt is
[`r2m-audio-vibration-payload-verification-20261120.json`](../../manifests/pretrained_reuse/r2m-audio-vibration-payload-verification-20261120.json);
its source download-index SHA-256 is
`715dd57fdf1967767f28f2b2246c57232dfba3e74ea548d46c83e68b882402d4`.
No media, extracted HDF5, feature arrays, or model weights were added to Git.

Identifier-only inventory covered 125 HDF5 members. It found 43,075 rows when
the five `multisensor-*` files are counted separately, corresponding to 8,615
rows in `multisensor-1` across all 25 archives, including the no-activity
sentinel archive. Each archive's five
members had matching row counts and per-participant counts; their exact
participant/trial arrays were confirmed identical for `01_walk`. To avoid
counting additional members as independent trial examples before their
measurement semantics are fully verified, this pilot uses only
`multisensor-1`. There are 14 real participant IDs and the publisher's `999`
no-activity sentinel. Inventory SHA-256:
`eb22c4f91149c03c580cd0b91bb85f371798dc3a31aba9ad69f3d09328e0fec0`.

The initial participant split was frozen before signal inspection. During a
subsequent schema probe, one audio and vibration row for sealed participant
`0` was inadvertently loaded to inspect the variable-length array shape. No
model, training, or validation score used it; only sample lengths and the
vibration rate were emitted. Nevertheless, participant `0` is now
quarantined from every split. The original split is retained only as a
superseded record and is not a blind audit split.

Before any fitting, a replacement split was frozen from the other 13 real
participants using the same seed-17 SHA-256 order, with 7/2/2/2 participants in
train/calibration/validation/sealed audit. It is recorded in
[`r2m-audio-vibration-split-v2-20261120.json`](../../manifests/pretrained_reuse/r2m-audio-vibration-split-v2-20261120.json),
manifest SHA-256
`0f1539f3ca4fa59fb854f62def1c538e1f5851576444b678af4aa050a393a6f0`:

| Split | Participant IDs | Unique trial rows in `multisensor-1` |
|---|---|---:|
| Train | 2, 4, 7, 8, 10, 13, 14 | 5,464 |
| Calibration | 12, 17 | 779 |
| Validation | 9, 15 | 775 |
| Sealed audit | 1, 16 | 578 |
| Quarantined | 0 | 788 |

The `999` sentinel is excluded. The publisher's no-activity archive contains
only that sentinel and contributes no trial-match examples. All source HDF5
members remained intact and were opened read-only.

## Signal-layout result

Before filtering or pairing, the replacement split was checked. A bounded
preflight read at most two records for each participant/activity in
train/calibration/validation, from `multisensor-1` only. Sealed participants
`1, 16` and quarantined participant `0` were not dereferenced by this
preflight. It read 364 records: 255 train, 44 calibration, and 65 validation.
Every selected audio array was nonempty and each selected vibration record
had three equal-length axes. The duration ratio
`(audio_samples / 16000) / (vibration_samples / recorded_rate)` ranged from
`0.9997295` to `1.0004018` (median `1.0000075`). The preflight artifact SHA-256
is `cb8302699037f1ecd8bb0f945981c6eee6e2ee963578649ab339e3a2502875b8`.

This supports a synchronized, same-trial pair construction on the inspected
records. It does **not** establish that trial identity is inferable from
cross-modal signal content, that same-activity hard negatives are sufficiently
matched, or that any model has useful accuracy. Rare validation activities
have only one trial and cannot support a within-activity negative; they must be
counted as uncovered by that comparison rather than silently removed.

The frozen participant/activity inventory gives 5,464 train trials with
cross-participant negatives available in all 24 active categories. Validation
has 775 trials; 21 categories have at least two trials, but only 10 categories
(582 trials) contain both validation participants and can support
cross-participant negatives. Calibration has 779 trials in 21 categories but
the two calibration participants share no activity category, so only
within-participant negatives are available there. Keep cross-participant and
within-participant validation strata separate; do not claim calibration on a
cross-participant task from this split.

## Next development gate

Build a deterministic, participant-safe pair index using `multisensor-1` as
the single fixed sensor position. Pair each positive with a different-trial,
same-activity negative matched on duration; prefer a different participant
when available. Freeze pair IDs and expose only train/calibration/validation
to fitting and model selection. Then compare a fixed temporal-correlation
baseline with the candidate-conditioned gated Decision model and explicit
shuffled, offset, and missing-stream controls. Keep sealed participants `1,16`
unopened until the candidate and pairing rules are frozen. Report the
quarantine and its reduced effective participant count in every result.

`PAYLOAD_HASH_GATE=PASS`; `IDENTIFIER_INVENTORY=PASS`; `REPLACEMENT_SPLIT=FROZEN`;
`BOUNDED_SYNC_LAYOUT_PREFLIGHT=PASS`; `PAIR_TASK_FEASIBILITY=PENDING`;
`MODEL_FIT=NOT_STARTED`; `SEALED_AUDIT_SIGNALS=NOT_ACCESSED`.
