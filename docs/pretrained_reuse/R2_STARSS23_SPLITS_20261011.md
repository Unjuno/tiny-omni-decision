# STARSS23 frozen room splits and training coverage

The previous probes established synchronized window decoding and actual frozen
feature extraction. This step freezes the next development comparison's room
assignment before model fitting or held-out label inspection. It is a real
inventory of the pinned source annotations, not a quality evaluation.

## Fixed assignment

Use only official dev-train recordings with video. Keep the two previously
inspected rooms (`tau:room12`, `tau:room13`) in train. Sort the remaining rooms
by SHA-256 of canonical JSON `[17,room]`; assign the first to calibration and
the next two to validation. All remaining rooms go to train. Every recording,
window, event query and option from one room inherits that assignment.

| Split | Recordings | Rooms |
|---|---:|---|
| Train | 49 | sony:room22; tau:room12,13,4,6,7 |
| Calibration | 8 | tau:room9 |
| Validation | 21 | sony:room21; tau:room14 |

The twelve missing-video recordings from room21 are excluded; other paired
recordings from that room remain eligible. All 78 eligible recordings are
assigned exactly once. The official dev-test contents remain unopened and are
outside these three splits. These development splits are not a sealed audit.
Global speaker identities are unavailable; no speaker-disjoint claim is made.

The exact lists and byte hashes are in `starss23-splits-v1/*.jsonl` and
`R2_STARSS23_SPLITS_20261011.json`. The result hash is
`209423b79c3bf1b98ce926aabf92a7f1f8262a5a8dea69e619f893fe9e1357b6`.
Neither model predictions nor event labels were used to assign rooms.
Calibration and validation annotation contents were not read by this operation.

## Training coverage: a material constraint

| Event | Train recordings containing event | Active 100 ms ticks |
|---|---:|---:|
| Speech (merged) | 47 | 62,321 |
| Clapping | 6 | 126 |
| Telephone | 11 | 664 |
| Laughter | 39 | 3,313 |
| Domestic sounds | 10 | 12,015 |
| Footsteps | 31 | 2,360 |
| Door | 22 | 855 |
| Music | 11 | 16,591 |
| Instrument | 3 | 3,270 |
| Water tap | 6 | 861 |
| Bell | **0** | **0** |
| Knock | 7 | 58 |

Ticks count distinct (recording, time, merged class), not duplicated concurrent
sources. They are annotation coverage, not consumed training examples or media
duration. Bell has no positive training examples under this frozen split. A
new supervised bell readout cannot learn positive discrimination from it.
Retain bell in reports, label it unsupported by positive training evidence,
and do not reshuffle rooms or remove the class to improve metrics. A future
pretrained label-based baseline may be informative but is a different method.

Only two validation rooms are available. Even many windows do not create more
independent environments; room-level uncertainty is severely underpowered.
An aggregate score dominated by speech or negative queries is insufficient.
Report all event classes with counts and balanced metrics plus NLL/Brier/ECE,
and compare fixed audio-only, video-only, joint and shuffled-video controls on
identical windows. A product-quality gate is still unresolved.

## Reproduction

Source base: `486fe46` plus `scripts/freeze_starss23_splits.py`.

```powershell
python -m scripts.freeze_starss23_splits --metadata C:\CodexArtifacts\pretrained-reuse-r2\starss23-source-audit-01\metadata_dev.zip --source-probe C:\CodexArtifacts\pretrained-reuse-r2\starss23-paired-smoke-01\result.json --output-dir C:\CodexArtifacts\pretrained-reuse-r2\starss23-room-splits-v1
```

The pinned source ZIP hash is checked before access. The output directory must
be new. Tests verify input-order determinism, room grouping, exposed-room
protection, missing-video/dev-test exclusion and that coverage reads train
annotations only. CPU suite: 268 passed, one skipped; Ruff passed. No weights,
media, original corpus or legacy run was changed.

Next execution needs additional paired media and features for these frozen
IDs. Given roughly 10.9 GB free disk, process one recording at a time and retain
features rather than duplicate complete extracted archives. Do not silently
restrict the comparison to the four earlier smoke windows.
