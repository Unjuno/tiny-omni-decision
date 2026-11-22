# NOAA MRMS / ASOS source screen — 2026-11-22

## Decision

**Status: `REVIEW_NOT_ADMITTED`.** The catalog metadata establishes that both
archives are available at useful temporal granularity, but it does not
establish independent storm groups, exact gauge-to-radar matching, or a
product-quality task. The downstream use terms for the mixed-origin ASOS
archive also remain unresolved. No observations, media, or model outputs were
downloaded or opened.

## Metadata-only measurement

A reproducible CPU-only script fetched the ASOS directory index for May 2024
and the MRMS `RadarOnly_QPE_01H` S3 object listing for 2024-05-01. It fetched
catalog responses only; `payload_objects_downloaded` is zero.

| Catalog | Listed coverage | Catalog-listed payload bytes | Receipt SHA-256 |
|---|---:|---:|---|
| ASOS one-minute Page 2, May 2024 | 873 station-month files / 873 station IDs | 3,353,725,870 | `80bf635975d4bcc6f80a7c50dcf45594b0ec845c0bbf329599cfda6115cabd01` |
| MRMS radar-only 1-hour QPE, 2024-05-01 | 720 timestamps, every 2 minutes | 471,875,988 compressed bytes | `05be978f9b034c93e69c2fdcbc8d95094f68b3d9a92798aa6179c815c3f5fc7b` |

These are catalog counts and listed object sizes, not downloaded data sizes or
independent sample counts. The NOAA product table describes the MRMS field as
a rolling one-hour accumulation and lists a two-minute cadence. Its public
data terms permit use with attribution and prohibit implying NOAA endorsement.
The NCEI ASOS metadata describes one-minute precipitation observations and
identifies NOAA, FAA, and DOD as originators, with citation and use-liability
conditions; this screen did not establish an explicit downstream license grant
for the mixed-source archive.

## Task fit and unresolved gates

The initial proposed three-way target—radar estimate greater than, within
tolerance of, or less than gauge amount—is directly computable from the two
input values by a deterministic comparison rule. By itself it would mostly
test whether the model learned that rule, not whether it reuses useful
multimodal representations. A future-precipitation decision task could be
more meaningful, but this metadata screen does not establish the needed
station/grid/time alignment, storm-event grouping, label thresholds, or
rights for such a task.

The 873 ASOS station IDs are not 873 independent weather events. The catalog
metadata does not show which stations overlap each MRMS grid/time sample,
station relocation history, or how to form storm-disjoint train/validation
groups. The single-day MRMS listing says nothing about the number of
independent storm groups. Therefore no split, feature cache, or task was
admitted.

## Scope and next gate

The script and tests are a metadata-screening aid only. Before reconsidering
this source, establish (1) explicit rights for the ASOS observations and any
underlying sources, (2) precise station/grid/time joins, (3) storm-event
grouping with disjoint splits, and (4) a task where both modalities carry
information beyond a trivial comparison rule. No data acquisition, feature
extraction, fitting, or evaluation is authorized by this screen.

## Sources

- [NOAA MRMS open-data registry and use terms](https://registry.opendata.aws/noaa-mrms-pds/)
- [NOAA/NSSL MRMS product table](https://www.nssl.noaa.gov/projects/mrms/operational/tables.php)
- [NCEI ASOS one-minute Page 2 metadata](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc%3AC00387/html)
- [NCEI ASOS product overview](https://www.ncei.noaa.gov/products/land-based-station/automated-surface-observing-systems)

The full machine-readable receipt is
[`noaa-mrms-asos-source-screen-20261122.json`](../../manifests/pretrained_reuse/noaa-mrms-asos-source-screen-20261122.json).
