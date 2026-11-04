# CONVERGE multimodal wireless source and task screen (2026-10-28)

## Decision

**Do not admit CONVERGE for this product-directed training phase.** The
official Zenodo README states that the dataset is licensed CC BY-NC-ND 4.0.
That non-commercial, no-derivatives license is not suitable for creating and
retaining a trained product candidate. No archive or media was downloaded, no
labels were loaded locally, and no training or evaluation was run.

## What the source provides

- Zenodo record `20849945`, version `v1`, DOI `10.5281/zenodo.20849945`.
- The official record describes synchronized RGB-D video, radio measurements,
  and annotations. Task 1 is a three-class blockage-state prediction target
  (`no`, `partial`, `full`) at a 142 ms horizon. Task 2 provides 3-D UE
  translation; Task 3 predicts future SRS channel measurements.
- Its README declares training scenarios `task1/exp1..exp5` and validation
  scenario `task1/exp9`, so the named scenario IDs do not overlap. It provides
  no test split. Whether the scenario IDs are independent collection groups
  beyond these identifiers is **UNKNOWN** until the source protocol is checked.
- The two archives are 3.2 GB (train) and 1.7 GB (validation), 4.8 GB combined.
  The current C: free-space snapshot was about 6.86 GiB, leaving roughly
  2.3 GiB before extraction. The extracted size was not inspected, so safe
  extraction capacity is not established.
- The Zenodo record's License field is blank in the rendered metadata, but the
  bundled official `README.md` preview explicitly says CC BY-NC-ND 4.0. Treat
  the README's more restrictive statement as controlling until the publisher
  clarifies otherwise.

## Scope and reproducibility

Read-only sources inspected:

- Record page: <https://zenodo.org/records/20849945>
- Official README preview:
  <https://zenodo.org/records/20849945/preview/README.md?include_deleted=0>
- Record file manifest: train ZIP MD5
  `57cb3b2b116289d81280c2147957343c`; validation ZIP MD5
  `98d8311c8a04311fd1f4a7b3b808dab0`; README MD5
  `2206cb87875da3fb4cfaad969b49820a`.

This is a source screen only. It does not establish model quality, input
complementarity, scenario independence, model-training rights beyond the
stated license, or permission for commercial/product use. Reconsider only if
the publisher provides terms that permit the intended model training and
downstream distribution, and a storage-safe, group-independent evaluation
protocol is available.
