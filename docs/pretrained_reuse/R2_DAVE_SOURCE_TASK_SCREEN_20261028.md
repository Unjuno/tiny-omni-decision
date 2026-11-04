# DAVE source/task screen — 2026-10-28

## Disposition

**Task fit: promising for a fixed AV diagnostic. Data admission: REVIEW/HOLD.**
DAVE is the strongest source found in this screen for measuring whether audio
and video are both used: its authors explicitly design the AV-alignment task
to require both modalities and provide matched visual-only, audio-only, and
text-only controls. It is evaluation-oriented, not a training corpus. No data
was downloaded or opened, no terms were accepted, and no model was evaluated.

Do not use DAVE until the upstream media/annotation terms and any access
agreement are resolved under the repository's source-admission policy. The
HF dataset card labels the packaged dataset MIT, while the authors' supplement
states that its inputs derive from sources with non-commercial and/or
agreement-bound terms. The GitHub README's broad “Everything is licensed under
MIT” statement does not establish rights to third-party media or annotations.
The underlying component terms govern this screen; the apparent MIT label is
not sufficient to admit the corpus for a product-aligned path.

## Pinned primary references inspected

- NeurIPS 2025 paper page: <https://proceedings.neurips.cc/paper_files/paper/2025/hash/4d5f03fdb238255019826032ae7cc8e2-Abstract-Datasets_and_Benchmarks_Track.html>
- Official DAVE repository: <https://github.com/gorjanradevski/dave>
- HF dataset card: <https://huggingface.co/datasets/gorjanradevski/dave>
- DAVE supplementary material: <https://openreview.net/pdf?id=4ZAX1NT0ms>
- Official Ego4D access documentation: <https://ego4d-data.org/docs/start-here/>

The NeurIPS abstract describes a benchmark designed to ensure both audio and
visual input are necessary. The DAVE card describes seven diagnostic tasks,
including AV alignment, sound absence, sound discrimination, visual-only,
audio-only, temporal ordering, action recognition, and audio classification.
The published AV-alignment example uses a video with an overlaid sound and
multiple-choice action descriptions. This is useful for controlled fusion
diagnosis, but is a constructed overlay task rather than proof of broad natural
AV decision quality.

The supplement says DAVE is constructed from:

- EPIC-KITCHENS, released under CC BY-NC 4.0;
- Ego4D, under its non-exclusive, non-transferable academic/research license;
- ESC-50, under CC BY-NC 3.0.

Ego4D's official access instructions require reviewing and accepting its
license agreement before obtaining data or annotations. The HF DAVE card
currently reports MIT and 17.6 GB total size; the viewer fails because it
contains a custom dataset script unsupported by the viewer. These facts do not
remove the upstream rights limitations or authorize accepting the agreement.

## Experimental value if later admitted

Use the AV-alignment subset as a fixed evaluation slice, not as training data,
and preserve DAVE's aligned visual-only, audio-only, and text-only controls.
Pre-register source/video grouping, option-order randomization, deduplication,
and calibration metrics before evaluation. The original package is roughly
17.6 GB, so a future run must first inspect exact relevant split/layer sizes,
disk headroom, revisions, and checksums; do not download the whole dataset by
default. Treat scores as DAVE-specific diagnostic evidence, not as the product
quality gate or as a replacement for a naturally recorded, rights-cleared
joint task.

## Next action

Keep DAVE `REVIEW` and continue screening sources whose component media and
annotations are explicitly permitted for the project's intended use without
executing a personal/institutional agreement. If none is found, the R2-M AV
quality gate remains open; do not substitute synthetic data or an unapproved
benchmark to claim closure.
