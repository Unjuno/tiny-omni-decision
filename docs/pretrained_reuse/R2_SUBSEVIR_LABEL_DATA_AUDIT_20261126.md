# sub-SEVIR label and task-data audit — 2026-11-26

## Scope

Read-only audit of the already extracted publisher **training** Zarr and the
existing frozen Path A manifest. No publisher validation/test payload, sealed
audit, model checkpoint, or feature cache was changed or opened for this audit.
The raw Zarr `.zmetadata` SHA-256 is
`cdcef87021e848a08f661e7caa30abb7a31810d7326f7c9a286c468d0523101a`; the
existing split manifest SHA-256 is
`3adcaf1713a562772ff44ba0bbf15533c411eb187699b96d0764139745e46b4b`.

## What the source actually represents

The official sub-SEVIR description calls this a tutorial subset for detecting
whether satellite/radar images contain lightning. The publisher tutorial's
Notebook 07 defines the `label_1d_class` classification task as “Does this
image contain a thunderstorm?”; Notebook 02 displays that class label
separately from `label_1d_reg` (flash count) and `label_2d_reg` (GLM lightning
map). Thus the project target is more precisely the next-frame
**thunderstorm-presence class**, not a direct prediction of lightning-flash
count. The exact source-side rule that generated `label_1d_class` has not yet
been traced. The tutorial subset reduces the original SEVIR data to 48×48
pixels and retains one continuous hour (12 five-minute frames) from each
four-hour event. It is useful for a narrow weather-sensor task, but it is not
a general video corpus or representative of the product's
image/audio/text/video decision tasks. See the [AI2ES source description](https://backup.ai2es.org/data/dataset-details-sub-sevir/),
the [tutorial repository](https://github.com/ai2es/WAF_ML_Tutorial_Part2),
its pinned [Notebook 07 at `e903c7c3`](https://github.com/ai2es/WAF_ML_Tutorial_Part2/blob/e903c7c3c22d8bc8c0a6058394bee21ae7eddca7/colab_notebooks/Notebook07_CNNs.ipynb)
(Git blob `80cbb958f68ccb27006d66b3369b2194d187804a`, SHA-256
`30fc6207d45ffe5dfe0da9cc79985d3f60e233985d49da2b819321db2c12c7be`),
and the [SEVIR source paper](https://papers.nips.cc/paper/2020/hash/fa78a16157fed00d7a80515818432169-Abstract.html).

The extracted training Zarr has 60,000 rows, four channels (`ir069`, `ir107`,
`vil`, `vis`), 5,000 event IDs, and exactly 12 rows per event. Its scalar
`label_1d_class` has 30,244 positive and 29,756 negative rows. The current
project-created task forms up to four windows per event: use the prior eight
five-minute frames to predict the next row's binary label. The fixed split
contains 3,969 train events / 15,828 windows and 998 validation events / 3,988
windows. Some possible windows are excluded because their timestamps do not
meet the exact five-minute cadence condition.

## Label persistence diagnostic

For the current validation targets, comparing each target label to the
immediately preceding row's `label_1d_class` gives **3,848/3,988 correct
(96.49%)**. An eight-row historical majority diagnostic gives **3,794/3,988
(95.14%)**. Across the full extracted train Zarr, 4,235/5,000 events have a
constant label over all 12 rows; 765 events contain at least one label
transition. In the validation targets, 3,848/3,988 labels repeat the previous
row.

These are **label-history diagnostics, not admissible model baselines**: the
current model receives four satellite/radar channels and does not receive
prior `label_1d_class` values. They must not be compared as if they had equal
information access. They do show that accuracy on this task is strongly
affected by label persistence and that a valid next experiment must include
an input-only persistence/comparison rule using only permitted preceding
sensor frames. The existing selected joint DINOv2+LSTM result (89.62% on the
same validation) does not establish that the representation beats a
task-appropriate sensor-only baseline, nor does the oracle label-history
score establish an achievable model ceiling.

The tutorial confirms the intended classification question but not the exact
source-side rule used to generate `label_1d_class` or how that class is
assigned over each five-minute row. Trace that data-generation rule before
interpreting this as a physical five-minute thunderstorm transition target.
That semantic check is a data gate, not a reason to adjust the model.

## Split and generalization limits

The current project split is a deterministic, label-independent hash split by
sub-SEVIR event ID, which prevents the same event ID from crossing train and
validation. It is **not** a chronological or geographic holdout. The
publisher's training subset spans many dates and the hashed groups from the
same calendar months occur in both partitions. The extracted Zarr does not
carry the full source catalog fields needed to verify NOAA episode identity,
event proximity, or spatial-neighborhood separation. Those forms of source
group overlap therefore remain unknown; zero exact event/frame overlap alone
does not establish independent weather regimes.

## Consequence for the next step

Do not add fusion gates, increase readout capacity, or interpret a high
accuracy on this task as broad model quality yet. First:

1. Trace the exact scalar label-generation code and establish what one row's
   label means in time and space.
2. Build a permitted-input baseline from the preceding sensor frames and
   report it beside the existing model on the already exposed validation,
   clearly marking the result as diagnostic rather than a new blind test.
3. For any new model selection, use a fresh group-disjoint validation carved
   only from the existing train groups; never reuse the exposed validation
   for tuning. Prefer a later, source-catalog-aware temporal/geographic holdout
   for the next admitted dataset.
4. Prioritize a rights-admitted, product-aligned dataset with sound labels,
   source-level split metadata, and enough independent groups before further
   architecture work.

No model was trained in this audit. The full Path A result and its limits remain
in [the original results report](R2_SUBSEVIR_PATHA_RESULTS_20261126.md).
