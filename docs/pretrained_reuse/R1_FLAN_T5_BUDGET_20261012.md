# FLAN-T5-small pre-download budget and compatibility audit

Pinned revision `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`.
[Publisher card](https://huggingface.co/google/flan-t5-small/blob/0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab/README.md)
declares Apache2.0; card is HF-authored, not a new independent rights opinion.

## Actual checkpoint structure versus local config instantiation

Downloaded metadata/config/card and only a bounded64KiB prefix of safetensors.
The22432-byte header describes **76,961,152** FP32 parameter elements with one
shared input embedding and a separate output lm_head. Full tensor payload was
not downloaded or loaded; advertised LFS hash is not a locally verified weight hash.
No inference, quality or runtime-compatibility success is claimed.

On installed Transformers5.6.2, `T5Config.from_dict` unexpectedly changes published
`tie_word_embeddings:false` to true. Meta-device model counting then gives
60,511,616, undercounting by16,449,536. Setting false manually still produces
separate input embeddings in this installed constructor, giving109,860,224 before
restoring the checkpoint's shared input structure. Neither is the checkpoint count.
This audit does not patch the installed environment or assert a safe runtime fix.
The initial attempt01 meta-only result is retained and superseded by02/header count.

## Product budget decision

- Current VideoPrism114,365,184 + Whisper-tiny37,760,640 leaves47,874,176 before
  text/fusion/heads under200M.
- Adding FLAN-T5-small makes229,086,976 before fusion/heads: fails the hard gate.
- DINO-small22,056,576 + Whisper + FLAN totals136,778,368, but replacing the video
  tower by framewise DINO has no proven temporal quality. Arithmetic alone is not
  a four-modality candidate. Do not silently remove temporal capability to fit.

**Do not download or adopt FLAN as a drop-in text upgrade in the current stack.**
This excludes one incompatible combination, not this model family universally.
A later alternative would need both a verified temporal representation and a
measured full-system budget. The existing smaller text path remains unresolved.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r1/flan-t5-small-budget-audit-02`.
All legacy weights unchanged; no GPU training, package upgrade or cloud cost.
Metadata/headers and reproducible audit script are saved. General95% remains unmet.
