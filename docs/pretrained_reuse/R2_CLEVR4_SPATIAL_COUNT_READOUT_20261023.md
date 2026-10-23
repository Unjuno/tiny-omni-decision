# R2 CLEVR-4 spatial count readout experiment

## Purpose and boundary

The fixed VideoPrism-B image candidate scored 60.44% on the count taxonomy of
the already observed official CLEVR-4 validation. That validation is closed to
selection and tuning. This follow-up compares a matched mean-pooled MLP with a
small option-conditioned readout over spatial tokens. It uses a new train/dev
split sampled only from the official training partition. It does not open the
official validation, a sealed audit, or any prior evaluation predictions.

This is a static synthetic image taxonomy task. Each image is repeated in the
encoder's eight frame slots, so this experiment does not measure temporal
reasoning or multimodal fusion. It addresses a specific spatial-count
bottleneck within R2 and does not close the roadmap's general multimodal gates.

## Frozen inputs and split

- Source: `sgvaze/clevr4@cddc78fb2a8359dc958987b2c750bfdd4bfd2c73`, CC-BY-4.0.
- Full source archive SHA-512:
  `7e8454f9f44d7b527eb20252a687b15b4cd28f3e9c79cf6147317971034c367b64bf6bba98d2a66f64b26f1e7eea2105a8acf221198342e89a844fd0007cc94c`.
- Source pool: the prior metadata-only 72,010-image official-train index. The
  old 8,424 train + 2,107 validation images and the separate 1,024/512 sample
  were excluded by both asset ID and media SHA-256 (10,531 unique identities).
- Seed 17 selects 30,000 train images and 5,000 development images by a stable
  SHA-256 ordering of `seed:image_id`; each image's four taxonomy questions
  stay together.
- Annotation JSON SHA-256:
  `1e51a1ea77da85c7c9203b422dcd3c337f053981f4bd6848be7c41f2784556cc`.
- Train ID-order SHA-256:
  `23e504526b0d9bbf1f8a901c1c22245aa765460dc7425b69fe3ace354be5018c`.
- Development ID-order SHA-256:
  `bc0434db01109ee1df5e135ea0989ac42d225cd756b638ad4c26a8b46323104c`.
- Source train index manifest SHA-256:
  `e48687af785a74864d2d4eb6e632a50d381d821a3ade8c2227eda7624647f410`.
- Source train labels SHA-256:
  `45de9d36d557619789dd6160e808f7ff0c531f06a298f0b2af80c9720103b695`.
- New train manifest / labels SHA-256:
  `feec0f9de345ab432d7e486fd0d9a8d58e7c92db6a07be1a2df4183ce96c474a` /
  `de1b66303a1a74545dbffc22cf857cb66cbc4bd81be9ae989a86167e2762b68f`.
- New development manifest / labels SHA-256:
  `100237bc05991db7cc4d7410f613d602e94ab3fe9a313752f30619ff74cf2a4b` /
  `7a97d68275bd8492748e6d2156d39affaddc9754fc0e0dfa5e16fcbd13266ded`.
- The four exclusion manifest hashes are captured in `split-summary.json`.
- The split manifests contain source archive member, byte length and SHA-256;
  every selected image was read from the pinned archive and checked against the
  indexed hash. Train/dev IDs and media hashes are disjoint.

The reproducible split is at
`C:\CodexArtifacts\pretrained-reuse-r2\clevr4-spatial-count-readout-attempt-02`.
Attempt 01 stopped immediately because the indexed manifest encodes split in
the labels file (`source_split`), not in the media rows. It produced no corpus
files. Attempt 02 applies that observed schema and is the frozen split.

## Matched candidates

Both readouts start from fresh initialization, use seed 17, the same 120,000
train questions, same 20,000 development questions, same shuffled question
order, eight epochs, AdamW at `1e-3`, weight decay `1e-4`, batch size 64, and
checkpoint selection by minimum macro development NLL across color, texture,
count and shape. The fixed development set is used for selection, so final
metrics are development evidence, not blind-test estimates. A fixed 2,048-image
training diagnostic is recorded separately from rolling minibatch CE.

| Arm | Visual input | Trainable parameters |
| --- | --- | ---: |
| Baseline | Mean of 64 spatial cells, then the existing 128-wide candidate MLP | 147,713 |
| Candidate | Option-conditioned attention over the ordered 64-cell grid | 311,553 |

The Candidate reshapes VideoPrism's `[8, 16, 16, 768]` output, averages only
the repeated-frame axis, then average-pools each 2x2 spatial neighborhood to
`[64, 768]`. Each of the 10 option embeddings independently queries those 64
cells. The scorer is permutation-equivariant to supplied options. No encoder or
projector is trained.

The fixed encoder has 114,365,184 parameters. The image-only resident totals
are therefore 114,512,897 for the baseline and 114,676,737 for Candidate,
before any future quantization. The 384-dimensional option vectors are reused
from the pinned MiniLM option cache; MiniLM is not resident at inference. No
ternary conversion is performed here.

## Execution state

The split preparer is complete. Unit tests cover deterministic exclusion,
duplicate media rejection, temporal-axis averaging, spatial ordering, scorer
output and option permutation equivariance (5 passed). Ruff passes for the
experiment files.

The CPU-only VideoPrism extraction path passed a 2-image-per-split smoke and a
32-image-per-split throughput run. The latter processed 64 images in 88.24
seconds including initial compilation; measured rates were 0.679 images/s for
train and 0.786 images/s for validation. At that observed rate the full 35,000
images are projected to take about 13.4 hours, with machine-dependent
variation. The smoke outputs are marked `smoke_complete` and cannot be used by
the trainer.

The matched trainer also passed a separate synthetic-fixture smoke using only
2 train and 2 development images. It completed both eight-epoch arms, wrote
distinct best/final checkpoints and predictions, and reproduced the selected
macro NLL after reloading the best checkpoint. Those fixture metrics are only
an execution-path check and are excluded from the model-quality results.

The full extraction is running as a separate CPU-only process in
`C:\CodexArtifacts\pretrained-reuse-r2\clevr4-spatial-count-readout-features-01`.
It reads only the frozen train/dev manifests and the archive, emits float16
`[N, 64, 768]` features to memory-mapped files, and validates every image hash.
The RTX 3080 remains occupied by an unrelated Orbitune process (15,821 MiB
reported), so this experiment does not request GPU memory. At the last hardware
check, system RAM had about 3.7 GiB free; the CPU extraction smoke peaked near
2 GiB working set. Disk had over 48 GiB free before extraction.

## Results and next gates

No readout training or fresh development metrics are reported yet. Once full
feature extraction is complete, the matched eight-epoch run will be started
from the immutable features. Results will include accuracy, NLL, Brier and
15-bin ECE by taxonomy and macro across taxonomies, fixed-train diagnostics,
selection epoch, predictions/probabilities, elapsed time and parameter counts.
The run will stop after the two planned arms; no follow-up will be selected by
reusing this development set without an explicitly recorded new hypothesis.

The old official validation's count score remains 60.44% and is not a measure
of this experiment. This work neither claims 95–98% quality nor supports a
general image-decision claim before fresh development results and independent
transfer tasks exist.
