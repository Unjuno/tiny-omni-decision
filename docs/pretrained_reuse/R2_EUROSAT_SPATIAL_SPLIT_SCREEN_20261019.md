# EuroSAT source and split screen — 2026-10-19

> Historical correction (2026-10-19): the initial screen below concluded that
> the RGB archive lacked a defensible split. A subsequent source audit found
> TorchGeo's pinned geographic split manifests, which map the RGB basenames to
> explicit longitude-based train/validation/test partitions. The initial
> conclusion was therefore too broad. The source-screen observations remain
> valid, but the decision and “no run” statement below describe only the state
> at the time of that first screen. A fixed readout experiment was later run;
> see [the spatial readout report](R2_EUROSAT_SPATIAL_READOUT_20261019.md).

## Decision

Do not use the RGB EuroSAT archive for a training/validation split in this
workstream. It is a plausible independent remote-sensing image task, but the
available RGB package does not expose the acquisition identity or coordinates
needed to prevent nearby or same-scene patches from crossing splits. A random
image split would therefore provide weaker evidence than the project requires.
This is a source-screen rejection for the RGB package, not a claim that EuroSAT
is unusable for every purpose.

## Source and local inspection

The official Zenodo v2 record describes EuroSAT as 27,000 labeled,
geo-referenced Sentinel-2 images in ten land-use/land-cover classes. The RGB
archive contains only the R/G/B bands as JPEG; the multispectral archive
contains 13 bands in GeoTIFF. The record points users to the Copernicus
Sentinel Data Terms and Conditions. The official author repository says the
dataset work is MIT-licensed and asks users to consider those Sentinel terms.

The official RGB archive was downloaded from Zenodo record `7711810` to the
external artifact directory
`C:\CodexArtifacts\pretrained-reuse-r2\eurosat-source-screen-01`.
Its size is 94,658,721 bytes. Published MD5
`f46e308c4d50d4bf32fedadd2d3d62f3b` matched. Local SHA-256 is
`b4f5b234ecb7d7ff9c6cdd046543b4717c53fd6e9815be6c0e80cc614f51b90`.

Read-only inspection found:

- 27,000 JPEG members across ten class folders, with published class counts.
- 27,000 distinct basenames and 27,000 distinct image SHA-256 hashes; no exact
  duplicate image bytes were found.
- Images are 64×64 RGB. The inspected JPEGs contain no GPS EXIF metadata.
- Filenames encode the class and a class-local integer only. The RGB archive
  does not provide coordinates, acquisition dates, source granule IDs, or an
  official split manifest.

The official repository issue asking for per-file acquisition dates and
coordinates confirms that the RGB files are numbered JPEGs and that the
corresponding metadata is not supplied there. The maintainer response points
to the multispectral GeoTIFF version for coordinates. Therefore a deterministic
hash split by RGB filename could prevent exact image reuse, but cannot establish
geographic group isolation or rule out related neighboring patches across
splits.

## Split policy and limitations

No split was created, no labels were used to select a model, and no training,
feature extraction, or quality evaluation was run on EuroSAT. In particular,
the sequential 60/20/20 split exposed by a third-party TFDS notebook is not
adopted: its ordering is not evidence of geographic grouping. We did not fetch
the 2.1 GB multispectral archive just to recover metadata because this would
not itself guarantee a clean grouping protocol or resolve the effort needed to
trace the RGB files to spatial footprints.

For a future EuroSAT task, first obtain a documented per-image mapping from
each RGB filename to source GeoTIFF and spatial footprint, verify that mapping
for all 27,000 files, then group by source acquisition/spatial block before
assigning train/validation/audit. If that mapping cannot be established, use a
different permissively licensed natural-image task with source-level groups
and a defensible official split. The RGB archive is preserved outside Git;
there are no dataset files or model artifacts in this commit.

## Sources

- [Official EuroSAT v2 record](https://zenodo.org/records/7711810)
- [Official EuroSAT author repository](https://github.com/phelber/EuroSAT)
- [Author issue on missing RGB georeferencing metadata](https://github.com/phelber/EuroSAT/issues/8)
- [Copernicus Sentinel Data Terms](https://cds.climate.copernicus.eu/licences/ec-sentinel)
