# CAV-MAE Sync source admission — 2026-10-13

Status: **REVIEW, not a runtime or quality pass**. Joint audio/visual pretraining
is relevant after independent frozen-feature fusion failed. No checkpoint,
dataset, or new training run was acquired in this audit.

## Evidence

- [Official repository](https://github.com/edsonroteia/cav-mae-sync/tree/0add2e7cbfcb2ff0315456b9db7ad347893ba54b),
  pinned to `0add2e7cbfcb2ff0315456b9db7ad347893ba54b`.
- README, BSD-2-Clause LICENSE, model source, retrieval entry point and weight
  download script are saved and SHA-256 recorded in the companion JSON.
- Local evidence: `C:/CodexArtifacts/pretrained-reuse-r1/cav-mae-sync-source-audit-01/`.
- The external Google Drive checkpoint's specific license scope is not
  established by these files. Repository software license is not automatically
  recorded as an explicit checkpoint grant. Do not promote to ALLOW yet.
- Upstream download script disables TLS verification. It was read, not executed;
  any eventual acquisition must retain certificate verification.

## Architecture accounting

The retrieved source uses eleven audio blocks, eleven visual blocks and one
shared block, width 768. The default retrieval model name disables the four
optional contrastive blocks, despite their constructor default being enabled.
Counting only attention/MLP weight matrices gives `23 * 12 * 768^2 = 162,791,424`.
With the existing 22,713,601-parameter text model, that lower bound is 185,505,025.
This excludes norms, biases, embeddings and all reconstruction modules; it is
**not a loaded-model count or proof of compliance**. A feature-only runtime may
fit 200M, whereas keeping reconstruction/optional blocks changes the budget.
No modules have been stripped, merged or substituted in existing models.

The source retrieval path uses a non-strict checkpoint load. A future local
runtime audit must enumerate missing/unexpected keys and verify all inference
weights rather than treating a successful call as load integrity. Existing
environment dependencies must be checked before importing upstream code.

## Next gate

Establish exact weight provenance/license, then count actual retained tensors and
verify inference/reload using a separate output directory. Only after those gates
would a fixed, approved-data AV comparison be justified. Retrieval results in
the upstream paper are not evidence of semantic decision accuracy, conditional
reasoning, or the project's 95–98% goal. Existing artifacts remain unchanged.
