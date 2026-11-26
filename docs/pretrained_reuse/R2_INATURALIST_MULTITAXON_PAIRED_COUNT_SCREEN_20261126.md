# iNaturalist multi-taxon paired media count screen — 2026-11-26

## Result

This count-only check does **not** admit a new image/audio task. The live
iNaturalist observations API currently reports 4,426 Aves, 484 Mammalia, 228
Amphibia, 7 Reptilia, and 0 Actinopterygii research-grade observations with
photo and sound both present and their observation/photo/sound license fields
all set to CC0. The Aves count is close to, but not identical with, the 4,410
results in the 2026-11-10 paired-media screen, confirming that the live API is
not a frozen corpus.

These are raw upper bounds before removing historically observed identities,
deduplicating photo/sound assets, grouping by observer, balancing taxa, and
checking whether the sound and photo actually concern the same individual or
event. Mammal and amphibian counts are already small before those reductions.
The local source-global identity ledger reports `history_status: INCOMPLETE`,
so it cannot certify zero overlap with all prior project exposure. Do not
train, select, or claim fresh generalization from this query. This screen
provides evidence to stop simple expansion of the prior bird-pair approach and
seek a better curated source/task.

## Query and receipt

- Endpoint: `https://api.inaturalist.org/v1/observations`
- Parameters: `quality_grade=research`, `photos=true`, `sounds=true`,
  `license=cc0`, `photo_license=cc0`, `sound_license=cc0`,
  `per_page=0`, `count_only=true`.
- Taxon IDs: Aves 3; Mammalia 40151; Reptilia 26036; Amphibia 20978;
  Actinopterygii 47178.
- Retrieval time: `2026-11-26T14:54:39.9840412Z`.
- The API returned zero observation rows; no media or labels were downloaded.
- External receipt:
  `C:\CodexArtifacts\inat-cc0-multitaxon-count-screen-20261126-attempt-01\summary.json`
- Receipt SHA-256:
  `39b9f32253b69f310f82d7d95c41dae891dc25b5ebc09f3266ed7f35cf51a08a`.

## Interpretation

The earlier 109-row bird candidate had only 92 train examples across 60
observer groups and 17 validation examples across 10 groups after the then
known historical-identity exclusions. The current broader non-bird counts
cannot remedy that weakness: they are live pre-filter counts, and the known
global identity ledger is incomplete. More importantly, paired observation
metadata does not establish that photo and audio show the same entity/event.
Therefore do not combine these rows into a larger class set or relax identity
gates. A future attempt needs a frozen source snapshot, sufficient unique
groups after all overlap filters, and a bounded manual modality-correspondence
audit before any model work.

The counts are feasibility evidence for this query only. They are not
population estimates, a frozen dataset, a zero-overlap certificate, or model
quality results.
