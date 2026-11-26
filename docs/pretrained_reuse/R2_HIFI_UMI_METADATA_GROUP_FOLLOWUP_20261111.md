# HiFi-UMI-2K episode-metadata follow-up — 2026-11-11

## Scope

This is a bounded source-screen follow-up to the 2026-11-03
[`HiFi-UMI-2K source screen`](R2_HIFI_UMI_2K_SOURCE_SCREEN_20261103.md).
It checked the current public repository revision and one episode-metadata
Parquet file only. No video, time-series observation/action payload, training
data, validation split, checkpoint, or sealed audit was opened. The Parquet
bytes were held in memory and not written to disk. No model inference or fit
was performed.

## Current pinned source state

The public Hugging Face API still reports revision
`a53b7b5784afdd50b2fda9195c9f724ef75ffdaf`, last modified 2026-07-29, 5,176
files, and `cc-by-4.0`. The publisher card exposes one `train` split. This is
the same revision already screened on 2026-11-03; it does not introduce a
publisher-provided validation/test split.

One file was read from that pinned revision:

- Path: `chunk-0000/part-0000/meta/episodes/chunk-000/file-000.parquet`
- SHA-256: `8f34934dbba0d6f325e3ac7cedb711fa74a2a6fd5671345e9dc2c1be25c8c436`
- Bytes: 1,870,339
- Rows: 1,125
- Read method: direct HTTPS response into memory; no local copy

Its columns include episode and task indices, task strings, row ranges, and
camera file/timestamp offsets. This sampled file has no scene, site, room,
subject, participant, session, or operator identity column. It contains 38
distinct task strings and 1,125 episode indices. The file's schema supports
episode-level grouping only; it does not prove that episodes are independent
scenes or sessions. This is a sample-part observation, not a claim that all
1,194 episode metadata files were scanned.

## Official paper follow-up

The authors' v1 paper (`arXiv:2607.25895v1`, Sections 3.3–4) states that
HiFi-UMI-2K is released under CC BY 4.0, that faces are masked before release,
and that the released subset contains 110+ scenes and 482,100+ episodes. The
paper describes scene-balanced exports and versioned quality metadata, but it
does not publish an episode-to-scene/session mapping or describe consent terms
for the captured people. Thus the public license is product-compatible as
stated and face masking is a documented safeguard; participant consent details
remain UNKNOWN. The sampled Parquet file still provides no key with which to
apply a scene-disjoint split. The paper's 110+ scene count is corpus-level
evidence, not evidence that a split can be reconstructed from released rows.

## Disposition

Keep `HiFi-UMI-2K` as `REVIEW_NOT_ADMITTED`. The CC BY 4.0 declaration is
product-compatible as stated and the paper documents face masking. However,
participant-consent details remain unknown, the release exposes only a train
split in the screened card, and the sampled episode metadata lacks a
defensible independent scene/session key. The underlying task supplies
continuous action trajectories; a frozen candidate-action Decision contract
and both-input necessity controls have not been specified or admitted. Do not
fetch media, extract features, or fit/evaluate from this candidate on the
current evidence.

The source remains a possible future lead only if the publisher supplies a
verifiable episode-to-scene/session mapping, a rights/consent basis for the
human-recorded video, and a task/split protocol with sufficient independent
groups. Do not infer a group split from episode index or task label.

## Reproduction environment

- Python: CPython 3.11 (local `py -3.11`)
- PyArrow: installed local version used to read the in-memory Parquet file
- Endpoint revision and file path: as above
- No package installation, GPU, cloud compute, or downloaded source/model
  payload written to the workspace

## Sources

- Authors' paper, pinned version: <https://arxiv.org/abs/2607.25895v1>
- Publisher dataset card at the screened revision:
  <https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K/tree/a53b7b5784afdd50b2fda9195c9f724ef75ffdaf>

