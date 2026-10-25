# iNaturalist natural-image Decision source screen (2026-10-26)

## Disposition

**PROMISING; not yet admitted.** This was an aggregate/API-metadata screen only.
No observation rows, coordinates, image bytes, or labels for individual
examples were retained; no data were downloaded and no model was trained.

iNaturalist may support a narrow but natural image-only Decision task: given a
bird photograph, assign probability across a caller-supplied set of species
IDs. The API can filter Research Grade observations and independently filter
the observation-data license and photo license. This separates explicit
CC0/CC-BY content from the default noncommercial material. The query returned
4,284,909 records for taxon Aves (`taxon_id=3`) when requiring Research Grade,
photos, and both the observation and photo licenses to be CC0 or CC-BY. A
species-count query returned 9,701 species. Among the most numerous species
under the same filters were Mallard, House Sparrow, Canada Goose, Rock Pigeon,
American Robin, Great Egret, Great Blue Heron, and Northern Cardinal.

These counts show candidate scale only. They are not an immutable corpus, do
not verify any individual image or its attribution, do not prove observer-
disjoint coverage, and say nothing about classifier performance. The live API
is not the experiment corpus.

## Rights and label gates

iNaturalist says licenses are set separately for observation metadata and
photos; the default for both is CC BY-NC. Its API exposes separate `license`
and `photo_license` filters. Only records whose observation and individual
photo are both explicitly CC0 or CC-BY should be considered. Preserve each
photo's own author attribution and license evidence; exclude any missing,
revoked, or conflicting record. Avoid CC BY-SA in this first bounded run to
keep model-derivative terms simple. Do not use coordinates, user names, or
location metadata as model features.

Research Grade is a community-consensus quality flag, not a guarantee of
taxonomic truth. The source says that grade normally requires agreement above
two-thirds at species level or lower, with additional DQA conditions. Before
admission, pin the data export and API response metadata; select one image per
observation; freeze a short species list by availability only; split globally
by observer ID (hashed for split construction); then reject cross-split
observation, image SHA-256, perceptual-hash, and normalized-label overlap.
Retain attribution and per-image license receipts outside public Git if they
contain personal display names.

## Source-use method and next gate

Counts were retrieved from the official iNaturalist API with
`quality_grade=research`, `photos=true`, `taxon_id=3`, `license=cc0,cc-by`,
and `photo_license=cc0,cc-by`. The API recommended-practices page says the API
is for small/medium batches, recommends about one request per second, and
directs bulk work to observation exports or the weekly GBIF research-grade
dataset. Any later acquisition must stay within those limits; a corpus-scale
download should use an approved export rather than crawl the live API.

The next gate is a bounded metadata-only preflight over a predeclared set of
species: verify sample counts, photo-level license fields, attribution fields,
observer diversity, split feasibility, and export availability. Do not fetch
media or fit a classifier until that gate passes. If admitted, evaluate one
fixed prequantized VideoPrism-B image feature path and a compact typed option
readout once on an independently observer-grouped validation set. Keep the
result task-specific and separate from the existing CLEVR-4 result. A later
sealed set must come from a frozen unused source snapshot and must not be used
for selection.

## Official evidence

- [iNaturalist licenses for observations, photos, and sounds](https://help.inaturalist.org/en/support/solutions/articles/151000175695)
- [iNaturalist photo and sound reuse guidance](https://help.inaturalist.org/en/support/solutions/articles/151000169918)
- [Official API license filters](https://www.inaturalist.org/api)
- [Research Grade definition](https://help.inaturalist.org/en/support/solutions/articles/151000169936)
- [API recommended practices and rate guidance](https://www.inaturalist.org/pages/api%2Brecommended%2Bpractices)
- [iNaturalist weekly research-grade GBIF export](https://www.gbif.org/dataset/50c9509d-22c7-4a22-a47d-8c48425ef4a7)
