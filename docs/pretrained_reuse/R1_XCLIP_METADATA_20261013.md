# X-CLIP prospective candidate audit — 2026-10-13

Purpose: investigate an already video-language-aligned representation after the
frozen motion/pooled-scene heads failed conditional reasoning. No training or
quality evaluation was performed in this audit.

Model:[microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32).
Pinned revision:a2e27a78a2b5d802e894b8a1ef14f3a8ce490963.
Downloaded config and model card only. Constructed Transformers XCLIPModel on
meta device from that config and counted actual module parameters:196,585,729.
This is a configuration-derived count, not verified checkpoint loading/parity.

Text63,165,952; vision122,909,952; temporal integration3,156,480;
prompt generator6,303,232; remaining projections/norm/logit scale comprise the rest.
Only3,414,271 parameters remain under the200M whole-product limit. Do not count
this as a compliant complete Omni model or add the existing audio/text encoders
without recomputing total residency. Removing temporal/text modules changes the
architecture and cannot preserve a claim of the original model's capability.

The Hub card marks MIT and describes supervised Kinetics400 training,8frames at
224px, contrastive video/text use. It explicitly says the card was authored by
Hugging Face rather than the original team. Upstream code/weight license details
and provenance need verification before weight acquisition/adoption. No inferred
commercial clearance from a metadata tag alone.

Local evidence:`C:/CodexArtifacts/pretrained-reuse-r1/xclip-metadata-audit-01/`.
Contains pinned config,card,SHA256 ledger,per-module counts and remaining budget.
No weights downloaded, GPU inference, benchmark evaluation or new fit.

Disposition: a potential comparison reference, with poor whole-product parameter
headroom. Do not launch another full-corpus run based solely on this audit.
A smaller video-language-aligned candidate or demonstrated shared-module reuse
would be needed to make the product budget credible. The completed trajectory
and scene experiments remain unchanged and not adopted.
