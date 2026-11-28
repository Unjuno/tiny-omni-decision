# Physion++ augmented-corpus coverage audit — 2026-11-28

## Scope

Read-only comparison of the local `physionpp-augmented-v1` corpus manifest and
run preflight against the frozen `fullclip-v1` train/validation manifests.
This checks whether the existing multimodal E-long corpus introduced new
Physion++ training examples. It does not load media, labels from sealed/test
sets, models, or checkpoints; it does not change the corpus or run a fit.

## Verified artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| augmented corpus manifest | 3,384 | `cdcc0a5a88856c1205a8448c32b37c729293b39f3a9f4e0f5fbc182d6bbea745` |
| augmented run preflight | 10,421 | `2e39bebb62c56f8bd8bb61007b6d9f8eab7cea9246820383fdb53ae1a9797c6b` |
| augmented train JSONL | 160,422,100 | `effef385e47b59c686de7837c3eaebbb07ec82462bf15b131889b8a2b1736c1e` |
| augmented validation JSONL | 10,327,766 | `bd9c75e362d1f82b2c467abd4fa45422daaa9d5a788d16ff2f035be0e7e1c6ba` |
| frozen Physion++ train JSONL | 860,532 | `0de1bbbb1cc54000936c1771ea54e375c214fedda36734536f31a44c4f64932b` |
| frozen Physion++ validation JSONL | 197,393 | `1254f01256197f1c2e23f8fa13dc03f87789c328f32bf768300134dd6b70d8dc` |

The augmented manifest's `physion_train_sha256` matches the frozen
Physion++ train SHA exactly. Its source counts show the same 652 Physion++
training records and 148 Physion++ validation records. Therefore the larger
98,840-row augmented train JSONL does **not** add Physion++ training videos;
it adds derived decision rows across modalities. The corpus has 8,455 video
decision rows (7,803 CLEVRER and 652 Physion++) but 811 unique video groups
in total (319 CLEVRER and 492 Physion++), so row count must not be reported as
unique video coverage.

The preflight records 98,840 train and 8,049 evaluation rows, with zero shared
source IDs, source assets, media identities, and normalized-content
fingerprints. Its 8,192-example planned consumption order reports zero repeated
examples, 5,755 unique underlying assets, and 811 unique video groups. These
are records from the preflight, not a new recount; the sealed audit was marked
not loaded. The CLEVRER question count is predominantly multiple questions on
reused scenes, not new underlying videos.

## Data conclusion

There is no safe additional Physion++ train coverage hiding inside this
augmented corpus. Treating the 98,840 derived decisions as 98,840 independent
training examples would overstate data scale. The video data comprises 319
CLEVRER and 492 Physion++ unique groups, with the existing Physion++
source-distribution, temporal-observability, and scenario-transfer
limitations documented separately.

Any actual data expansion must come from a newly admitted, rights-cleared
source or from publisher-designated training records not already present in
the frozen corpus. Preserve all validation/test/sealed partitions; verify
underlying video/scene grouping, normalized question/option overlap, label
observability, source balance, and media hashes before fitting. Do not
relabel rows, move validation into training, or infer improvement from a
larger derived-row count.

## Provenance and limits

- Augmented corpus:
  `C:\\CodexArtifacts\\tqpp\\physionpp-augmented-v1\\corpus`
- Frozen corpus:
  `C:\\CodexArtifacts\\tqpp\\fullclip-v1`
- `run-preflight.json` names an output path under `physionpp-augmented-v1`,
  but that run directory was not present at the path checked. This does not
  establish whether another copy of any run artifacts exists elsewhere; no
  training completion is inferred from this audit.
- No original corpus, preflight, run, media, checkpoint, or feature cache was
  modified.
