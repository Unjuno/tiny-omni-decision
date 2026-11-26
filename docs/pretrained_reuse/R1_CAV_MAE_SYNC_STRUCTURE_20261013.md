# CAV-MAE Sync CPU structure check

This is an **untrained structure check**, not a pretrained runtime/quality pass.
Pinned source and rights status remain in R1_CAV_MAE_SYNC_ADMISSION_20261013.md.

`python -m scripts.audit_cav_sync_structure` constructs the upstream class on CPU
with audio length 1024, width 768, eleven modality-specific layers, eight register
tokens, CLS enabled, four-frame setting and contrastive heads disabled. It calls
`forward_feat` with a batch of two zero audio/image inputs. It does not exercise
the temporal batching/preprocessing in the retrieval driver.

| Measured instantiated tensors | Parameters |
| --- | ---: |
| Full upstream class including reconstruction | 191,049,984 |
| Parameters used by feature path (hooked modules plus direct root tensors) | 164,374,272 |
| Feature path plus existing 22,713,601 text encoder | 187,087,873 |

All four outputs were finite; shapes were `[2,512,768]`, `[2,196,768]`, and two
`[2,768]` CLS outputs. No weights were loaded, trained, saved or quantized.
The full class plus text exceeds 200M. The feature path leaves approximately
12.9M parameters for the rest, if unused modules are actually removed and all
retained weights pass checkpoint loading. The current full object still retains
191M parameters: the hook count is not a claim of implemented runtime pruning.

First attempt using an existing isolated timm dependency failed at construction:
`Attention.__init__() got an unexpected keyword argument 'qk_scale'`.
Second attempt used upstream-requested timm 0.4.5 installed with `--no-deps` into
`C:/CodexArtifacts/pretrained-reuse-r1/cav-mae-sync-deps-01/`. The working Python
environment was not upgraded. Torch remains 2.6.0+cu124. The successful full
tensor inventory and source hashes are in
`C:/CodexArtifacts/pretrained-reuse-r1/cav-mae-sync-structure-02/result.json`.

Next gate remains exact pretrained weight provenance and integrity, followed by
real preprocessing/inference and fixed task evaluation. Code/license inspection
does not establish weight licensing, accuracy or usable temporal reasoning.
Also, the original CAV-MAE repository's default branch is **master**, pinned at
`68fe8c2a3917dc2926e41f796bfdcb331a64b42c`; its old `main` branch is not current
release evidence. No product adoption follows from this structure check.
