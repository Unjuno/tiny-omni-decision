# R2 PAMAP2 sensor admission screen — 2026-10-21

Status: **source screen only; not admitted for a MOMENT-based quality run**.
No dataset archive was downloaded or opened, and no model inference or fitting
was performed.

## Source facts

- Official source: [UCI PAMAP2 Physical Activity Monitoring](https://archive.ics.uci.edu/dataset/231/pamap2%2Bphysical%2Bactivity%2Bmonitoring), DOI `10.24432/C5NW2H`.
- The UCI record states CC BY 4.0, 3,850,505 timestamped rows, 9 subjects,
  18 activity labels, three body-worn IMUs (wrist, chest, ankle) and heart
  rate. Sensor streams have different rates; raw records contain missing
  values and the IMU orientation fields are invalid. The listed source archive
  is 656.3 MB.
- The UCI record establishes a dataset license, but it does not establish an
  immutable downloaded-file hash in this audit. Source checksum and archive
  contents therefore remain unverified.

## Pretraining overlap and statistical limits

MOMENT's [official project page](https://moment-timeseries-foundation-model.github.io/)
and [Time-series PILE card](https://huggingface.co/datasets/AutonLab/Timeseries-PILE)
describe pretraining over multiple public time-series collections, including
UCR/UEA, but the audited public materials do not provide a complete
dataset-by-dataset training manifest sufficient to prove PAMAP2 was excluded.
PAMAP2 overlap with MOMENT is therefore **UNKNOWN**. This is not evidence that
PAMAP2 was used; it is insufficient evidence to treat a MOMENT score on it as
clean transfer. Do not use PAMAP2 to select or compare MOMENT-based candidates
until exact pretraining provenance rules out overlap.

The 9-subject source also cannot support a high-powered subject-generalization
or release-quality claim. Randomly splitting overlapping windows would leak
subject and adjacent time-series content and is prohibited. Any future
low-power diagnostic would need subject-disjoint outer folds, all preprocessing
fit within each training fold, per-subject results and explicit uncertainty;
it would remain exploratory regardless of its window-level Accuracy.

## Disposition

- License: **source metadata says CC BY 4.0**, attribution required.
- Exact archive hash/content audit: **not performed**.
- MOMENT overlap: **UNKNOWN**; no MOMENT evaluation or selection authorized.
- Independent-group scale: **insufficient for a release-quality sensor task**
  (9 subjects).
- Training/evaluation in this screen: **none**. No weights, metrics, or
  existing run artifacts were changed.

PAMAP2 is not the next experiment. R2 sensor work remains optional and
research-only; first seek a sensor task with enough independent groups and a
pretrained encoder whose exact training sources can be audited. If none is
available, preserve the current sensor result as low-power research evidence
instead of claiming a general sensor capability.

## References

- UCI source record and license: <https://archive.ics.uci.edu/dataset/231/pamap2%2Bphysical%2Bactivity%2Bmonitoring>
- MOMENT pretraining source summary: <https://moment-timeseries-foundation-model.github.io/>
- Time-series PILE card: <https://huggingface.co/datasets/AutonLab/Timeseries-PILE>
- Existing MOMENT model/data overlap audit for UCI HAR: `R1_MOMENT_SENSOR_20261011.md`.
