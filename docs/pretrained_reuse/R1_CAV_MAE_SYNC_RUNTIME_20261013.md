# CAV-MAE Sync pretrained CPU smoke

Official checkpoint was acquired with TLS verification enabled from the Google
Drive ID in the pinned upstream download script. Research inspection only;
weight license scope remains REVIEW and no product adoption is made.

- Bytes: 762,886,272; SHA-256:
  `a035b04cf615fe4c85d6aa1e74b26d8223a85945957664e66abea16e183629bd`.
- Acquisition: `C:/CodexArtifacts/pretrained-reuse-r1/cav-mae-sync-weight-audit-01/`.
- CPU load uses `torch.load(..., weights_only=True)`; all 650 keys have the
  `module.` prefix removed, then `load_state_dict(strict=True)` succeeds.
- Source revision: `0add2e7cbfcb2ff0315456b9db7ad347893ba54b`.
- Upstream retrieval's `4s` configuration uses audio length **416**, not the
  constructor default 1024 used in the previous untrained structure check.
  Audio positional embeddings in the actual checkpoint are `[1,208,768]`.
  No positional interpolation or dropped/missing weights were used.

| Actual loaded configuration | Parameters |
| --- | ---: |
| Full class, including reconstruction | 190,660,864 |
| Feature path used tensors | 164,140,800 |
| Feature path plus existing text encoder | 186,854,401 |

The feature count is measured by module hooks plus directly used root parameters.
Unused modules have not yet been removed from the full object. Full class plus
text exceeds 200M; the smaller figure is not a packaged runtime claim.

`python -m scripts.audit_cav_sync_runtime` runs on CPU with isolated timm0.4.5 and
Torch2.6.0+cu124. Two zero audio/image inputs produce finite local and CLS outputs;
an identical second call is bitwise equal. Shapes: `[2,208,768]`, `[2,196,768]`,
and two `[2,768]`. Results and tensor inventory are saved under
`C:/CodexArtifacts/pretrained-reuse-r1/cav-mae-sync-runtime-01/result.json`.

This verifies pretrained loading and basic execution only. It does not validate
real-media preprocessing, temporal batching, semantic AV correspondence, decision
accuracy or calibration. No training, validation selection, sealed data access,
or modification of existing artifacts occurred. Next: real-input preprocessing
and fixed task evaluation, with product rights gate retained.
