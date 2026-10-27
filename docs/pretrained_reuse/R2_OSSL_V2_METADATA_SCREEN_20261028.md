# OSSL-v2 audio/video source: metadata screen

The public [OSSL-v2 dataset card](https://huggingface.co/datasets/McAuley-Lab/OSSL-v2)
describes paired film clips and music, declares CC-BY-4.0 for the dataset and
claims movie-level split isolation. Its [paper](https://arxiv.org/abs/2608.11576)
describes public-domain film sources. Dataset-level statements do not by
themselves verify rights to every film, soundtrack or derivative in the
intended jurisdictions. This is a **REVIEW** candidate, not admitted media.
The card describes music generation, not independently verified labels requiring
both modalities for a Decision task; dialogue, source separation and synthetic
annotations need separate examination. No audio, video, test split file,
checkpoint or model was accessed.

### Audio semantics from the upstream paper

The upstream paper states that the paired audio is a **source-separated music
stem**, selected using an automatic music-event detector. It also notes that
source separation can leave non-music residuals. Its experiments separately
extract dialogue from movie audio; the paper's description does not establish
that the distributed WAV is an unmodified full-scene audio track, nor that a
visible object caused the soundtrack. Thus these paired files cannot be assumed
to supply speech or visible-source sound evidence for a joint Decision task.
Any proposed soundtrack-mood or correspondence task would need its own label,
audio provenance, time-alignment and shortcut controls. The paper's
[dataset-construction section](https://arxiv.org/html/2608.11576) is the source;
no sample audio or video has been opened for this screen.

Metadata snapshot: revision `3ed5a37d933084e5ec134e2138b8faa156cf021b`,
`metadata.csv` size 1,220,772 bytes, SHA-256
`0e7bdd7b92c29e9117a635ebe40d29ed6d82d04c60df5b3412e1d0bc728b4cbf`.
The file has 1,812 unique movie IDs; 1,655 train movie rows represent 31,011
clips and 1,630 distinct YouTube IDs. Eighteen train YouTube IDs occur in
multiple movie rows, covering 43 rows. **Three YouTube IDs occur across the
supplied train and test movie rows.** This verifies identifier overlap, not
whether those sources contain duplicate visual or audio segments. Movie-ID
disjointness alone is insufficient for a source-independent split. Metadata
contains no per-asset license field or proof of clip-level rights.

The compact output is
`C:/CodexArtifacts/pretrained-reuse-r2/ossl-v2-metadata-20261028-attempt-01.json`,
SHA-256 `d50aa9a8c82c50b409f836611d174f03a889b457e0e070769696a68bb4a7aedf`.
`scripts/audit_ossl_metadata.py` checks the pinned payload and saves aggregate
counts; two CPU tests and Ruff passed. Before any use, resolve source-identity
collisions and verify each selected asset's provenance and rights, define a
task with meaningful complementary inputs, and freeze group-disjoint splits.
