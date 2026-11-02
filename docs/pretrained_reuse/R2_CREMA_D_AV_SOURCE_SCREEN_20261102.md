# CREMA-D audio/video source screen — 2026-11-02

## Disposition

**Candidate for a narrow AV-perception diagnostic only; not admitted for
training or product evaluation.** No audio/video payload, labels, or annotations
were downloaded or opened. This screen uses public source documentation only.
It does not supersede the failed STARSS23 AV task or supply a product task.

## Pinned upstream evidence

- Official project repository:
  `https://github.com/CheyneyComputerScience/CREMA-D`
- Repository revision: `1658cd342dff90010aa843eaeebd53610a08b1dc`
  (2025-03-21).
- `LICENSE.txt` blob `941b7e9eb2ccbc86efb34d5d2f94a1e744d2e56a`, 312 bytes,
  SHA-256 `e76747a4efa4b338914bb3f930b5d3d3c3ac15b886bbabf897f6a19731f987f8`.
- `README.md` blob `394e5758ef79b060a53fc5d083a017b552598d84`, 13,316 bytes,
  SHA-256 `bf3d3a070bf3dfb8eaceb0378fdb007388796bd24924e71fb17d0f65564136da`.
- The README identifies the database license as ODbL 1.0 and states that
  individual contents use the Database Contents License (DbCL 1.0). Therefore
  the database-level license alone does not settle all rights in face/voice
  recordings, derived model artifacts, or privacy/consent constraints. The
  README also asks repository users to complete a usage-registration form.
  No form was submitted, no rights were accepted on the user's behalf, and no
  content files were fetched.

## Task fit and risks

The authors report 7,442 clips from 91 actors, 12 repeated sentences and six
acted emotion categories. The same clips have separate audio-only,
visual-only, and audiovisual crowd ratings. The paper reports human recognition
of intended emotion at 40.9% audio-only, 58.2% visual-only and 63.6% AV; these
are human benchmark results, not model scores. The design can support a narrow
same-target AV-complementarity study if the raw ratings and clip identities are
usable, but acting, repeated utterances, ambiguity and annotator disagreement
limit the task. It is a poor proxy for general-purpose action choice or
95–98% world-state Decision quality.

Actor-disjoint grouping is available in the study design and TFDS metadata, but
TFDS exposes only the audio stream. A usable AV split would need a frozen
speaker mapping for the original AV files and a sentence-aware leakage audit;
repeated sentence templates cross actors. Intended-emotion labels must not be
silently substituted for human-perceived labels, and the three rating
conditions do not share the same target distribution without a predeclared
common target. The expressive video/audio contains identifiable people, so the
source's consent and content-rights scope need review before use.

## Decision

Do not use CREMA-D for the current candidate. Reconsider it only if the
repository access request and content-rights/consent scope are resolved, a
reproducible actor/sentence grouped AV split is established from metadata, and
a single human-perceived target is frozen before any model evaluation. Even if
those gates pass, report it only as a narrow AV emotion benchmark, never as
evidence for the product's broad Decision target. No model was trained, no
GPU was used, no split or quality metric was created, and no sealed evaluation
was accessed.

## Sources

- [Official CREMA-D README and license statement](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/README.md)
- [CREMA-D source paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/)
- [CREMA-D dataset card](https://audeering.github.io/datasets/datasets/crema-d.html) (audio-only mirror; reports ODbL and commercial use)
- [ODbL 1.0 official summary](https://opendatacommons.org/licenses/odbl/summary/)
