# R2 Open Screen Soundtrack Library v2 source screen — 2026-10-22

## Decision

**REVIEW ONLY; do not download media, train, or count OSSL-v2 toward product
quality.** The dataset offers a plausible video-conditioned soundtrack choice
task and a train split grouped by movie, but the underlying film and soundtrack
rights are not independently established at item level. It is also not yet
shown that the proposed decision task requires both audio and video.

## Public metadata and task fit

The dataset card describes 34,343 audio-video clip pairs from 1,812 movies,
with 31,011 train clips and 3,332 test clips. It states that movie IDs do not
cross train/test and that clip identifiers encode movie identity. The repository
is labeled CC BY 4.0 and the complete dataset is listed as 195 GB. The paper
describes source-separating soundtrack music and selecting clips from public-
domain films downloaded from YouTube.

A future task lead is: given a short silent or muted film clip and four
soundtrack candidates, select the soundtrack that belongs with the scene.
This could test contextual audiovisual matching, but a paired clip alone does
not prove both modalities are needed. Any evaluation would need movie-disjoint
groups and fixed candidate sets, plus audio-only, video-only, shuffled-video,
candidate-order, and hard-negative controls. The task should be treated as
soundtrack matching, not general audiovisual reasoning.

## Bounded train-manifest check

Only the small public `splits/train.txt` ID list was fetched, at dataset revision
`3ed5a37d933084e5ec134e2138b8faa156cf021b`. No media or label payload was
downloaded. The manifest contains 31,011 rows and 1,655 distinct movie groups;
clips per movie have minimum 1, median 12, and maximum 339. Its SHA-256 is
`0f0a23fa9e3558e97fff46e3c1a9cb5d8dffbbc8aa52159aac8c9b5dcf5baba4` and its
local size is 626,328 bytes. Receipt: `C:\CodexArtifacts\pretrained-reuse-r2\ossl-v2-train-manifest-attempt-01\receipt.json`.
The HTTP Content-Length value is UNKNOWN because response-header parsing failed;
the local byte count above was measured from the saved file.

During metadata inspection, the public Hub tree API returned file-path metadata
including paths for the publisher's test split. No test IDs, labels, or media
payloads were selected or read. Because the public test split's path metadata
was exposed during this screen, do not treat that publisher test split as a
sealed or blind audit set. Do not access it further. If the source is later
admitted, form any new evaluation only from untouched movie groups in the
permitted training pool, with pinned inputs and a documented isolation gate.

## Rights and provenance

- The Hugging Face card labels the dataset CC BY 4.0, and its paper describes
  the source films as public domain. However, the per-film public-domain basis,
  soundtrack/music rights, and any item-level exclusions were not independently
  verified. YouTube as an acquisition path does not itself establish those
  rights.
- The paper describes separated soundtrack stems and event-selected clips;
  source-film and derived-audio provenance needs to be traceable per asset.
- No media, test payload, or full dataset was downloaded. The 195 GB corpus is
  outside the current local storage budget and is not needed for this screen.

Before reconsidering, require a per-film rights/provenance manifest, confirm
commercial model-training and derivative-weight permissions, verify exact
audio/video alignment and item licenses, and pre-register a both-required task
with leakage-resistant movie grouping and unimodal/shuffle controls. Keep this
as a task/source lead until those gates pass.

## Evidence inspected

- Dataset card and split description: <https://huggingface.co/datasets/McAuley-Lab/OSSL-v2>
- Paper: <https://arxiv.org/abs/2608.11576>
- Pinned dataset revision: `3ed5a37d933084e5ec134e2138b8faa156cf021b`

No training, feature extraction, model evaluation, or GPU work was performed.
The existing Orbitune GPU process was left untouched.

## Roadmap effect

OSSL-v2 is a promising AV matching task lead but is not an admitted source and
does not close R2's both-required quality gate. Current blockers are item-level
rights/provenance, local storage, and unproven dependence on both modalities.
The publisher's test split is excluded from any blind-audit claim due to the
test-path metadata exposure recorded above. Existing model, corpus, checkpoint,
and artifact files remain unchanged.
