# STARSS23 full development corpus: acquisition and audio features

Source code: commit `4a741d0` on `docs/pretrained-reuse-roadmap-20261010`.
The adjacent JSON records exact external report hashes and compact results.
Large media, per-window records and feature arrays remain outside Git under
`C:/CodexArtifacts/pretrained-reuse-r2/`.

## Completed acquisition

`starss23-frozen-media-ipv4-02` completed all 78 synchronized pairs (156 assets).
The 14 verified complete pairs from the interrupted first acquisition were reused;
original artifacts were preserved. The second acquisition took 9,998.91 seconds.
Exact HTTP ranges, ZIP member CRC and saved-file SHA-256 checks were retained.
Whole-archive MD5 verification was not performed because acquisition used ranges.
Official dev-test contents were not accessed.

## Frozen windows

`starss23-all-windows-v1` passed the full acquisition gate and generated all
complete, nonoverlapping five-second intervals, without selecting by labels.
Partial tails are recorded individually in the external report.

| Split | Windows | Binary event queries |
| --- | ---: | ---: |
| Train | 1,955 | 23,460 |
| Calibration | 279 | 3,348 |
| Validation | 486 | 5,832 |
| Total | 2,720 | 32,640 |

The frozen room split is unchanged. Calibration and validation development
annotations were used to construct the predefined event queries. Bell has zero
positive training queries; that class remains in the fixed task. Validation has
only two rooms, limiting uncertainty estimates and generalization claims.

## Completed audio extraction

`starss23-audio-corpus-features-v1` exited successfully and reports PASS for all
2,720 windows from 78 recordings. Frozen EfficientAT DyMN0.4 produced a
`2720 x 384` float32 feature array on CPU, with saved-array reload verification.
Total wall time was 174.44 seconds; peak working set was 604,585,984 bytes.
The report records the exact source, checkpoint, runner and manifest hashes.

VideoPrism corpus extraction is still running in the existing CPU environment.
No full video extraction success, readout training result or joint-input quality
gain is claimed here. After video extraction passes, the fixed preregistered
audio-only, video-only and joint readout comparison can execute using these same
windows. Existing Teacher/QAT artifacts and sealed evaluations were not changed.
