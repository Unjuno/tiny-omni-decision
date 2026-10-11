# Pretrained candidate audit (R1)

Snapshot date: 2026-10-10. These are feasibility records, not task-quality claims. A successful tensor forward proves loading and numerical execution only.

## Executed local candidates

| Role | Exact model/revision | Rights and provenance | Loaded parameters | Artifact bytes | Result |
|---|---|---|---:|---:|---|
| Image/video | Meta V-JEPA 2.1 ViT-B encoder; official code `facebookresearch/vjepa2@204698b45b3712590f06245fbfba32d3be539812`; checkpoint `vjepa2_1_vitb_dist_vitG_384.pt`, SHA-256 `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d` | Pinned repository contains MIT and Apache license files; README says most of V-JEPA 2 is MIT, lists three Apache-licensed source files, and links the hosted checkpoint. It does not explicitly name a license for the external weight file; rights status **REVIEW**. | 86,833,152 encoder parameters; training predictor excluded from inference count | 347,332,608 encoder tensor bytes; original training checkpoint 1,664,223,428 bytes | Official factory strict-load and independent strict EMA load passed; image and 8-frame video outputs matched exactly. Technical pass; product rights gate open. |
| Japanese text | `cl-nagoya/ruri-v3-30m@24899e5de370b56d179604a007c0d727bf144504` | Model card Apache-2.0; base model `cl-nagoya/ruri-v3-pt-30m`. The card identifies Japanese as the supported language. | 36,705,536 | 146,828,152; SHA-256 `e94155e342cbfc33280c345c6e2912905fb5de6e1a3adeb938f395aafd2f8832` | Local tokenizer/model load and finite Japanese and English text tensors passed. English forward is a plumbing check only; English quality is unsupported/unmeasured. |
| English text | `sentence-transformers/paraphrase-MiniLM-L3-v2@4ca70771034acceecb2e72475f72050fcdde4ddc` | Model card Apache-2.0. | 17,389,824 | 69,569,488; SHA-256 `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433` | Local tokenizer/model load and finite [2,384] mean-pooled vectors; task quality unmeasured. |
| Audio | `MIT/ast-finetuned-audioset-10-10-0.4593@f826b80d28226b62986cc218e5cec390b1096902` | HF model card states BSD-3-Clause; original AST repository is BSD-3-Clause and publishes AudioSet weights. The HF team authored the card, not the AST authors. | 86,594,063 | 346,404,948; SHA-256 `ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d` | Local processor/model load and finite 527-class output passed on a 0.2-second fixture. Represents environmental sound tagging, not speech understanding. |
| Audio speech | `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af` | Model card Apache-2.0. | 37,760,640 | 151,061,672; SHA-256 `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395` | Frozen encoder returned finite [1,1500,384] features on zero-audio plumbing input; speech quality unmeasured. |

The initial task-matched four-tower proposal (V-JEPA + Japanese Ruri + English MiniLM + Whisper) is **178,689,152 parameters / 714,791,920 FP32 weight bytes**. It fits the parameter target and fails the 200MB package target before compression. These are alternate language/audio paths, not evidence of multilingual or speech Decision quality. A machine-readable record of rights, weight hashes, parameter counts, load evidence and runtime status is in [`R1_CANDIDATE_INVENTORY.json`](R1_CANDIDATE_INVENTORY.json); null rights values mean unresolved, not permission.

The three executed towers total **210,132,751 parameters** and **840,565,708 resident weight bytes** at the measured FP32 storage. This exceeds the 200M parameter target by 10,132,751 (5.1%) and the 200MB package target by 640,565,708 bytes. The V-JEPA source checkpoint also contains a training predictor that is not part of this deployment tally. No quantization or conversion was performed.

## Inspected but not selected for the first deployable path

| Candidate | Evidence | Disposition |
|---|---|---|
| EfficientAT `dymn04_as` | Pinned source `fschmid56/EfficientAT@a425fdce92572e602a1d5634799bd9f1f2efa806`; official `v0.0.1` release asset 8,086,973 bytes, SHA-256 `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`. Code is MIT; upstream owner explicitly permitted commercial checkpoint use in issue #44 on 2026-09-07. Strict load and local 2-second forward passed: 1,961,239 parameters, finite 527 logits; exact details in `R1_RELEASE_LICENSE_FOLLOWUP_20261011.md`. | ALLOW as an environmental-audio candidate. This does not establish speech keyword or joint audio-video Decision quality. |
| TencentARC ViT-Lens-L audio | HF repo revision `a80fe8f1a4039747f4e9a3e40d5287607ef940d3` advertises Apache-2.0; `vitlensL.pt` is 8,891,845,036 bytes and `vitlensL_audio.pt` is 3,434,449,193 bytes. Published audio results used a CLIP ViT-L-14 backbone. | Not downloaded: the pair is over 12.3GB before runtime code and far beyond the 100–200M product target. It is a separate original-backbone reference, not a V-JEPA-compatible Lens. |
| Meta PE Core T/16, S/16, B/16 | Official docs list T/16 (10M vision + 40M text), S/16 (20M + 40M), and B/16 (90M + 310M) checkpoints and state released weights are Apache-2.0. PE Core model pages list the weights, but anonymous Hub API access returned HTTP 401 in this audit; exact S/B revisions, hashes, sizes, and local loads remain unknown. The repository `LICENSE.PE` is Apache-2.0, while package setup metadata separately declares FAIR Noncommercial Research. | Pretrained weights are listed; access status is UNKNOWN for this audit. Separate weight and implementation terms. No product use of the official implementation until its terms are resolved or a license-clean implementation path is established. |
| Meta PE-AV small | HF revision `dd050762bb9704ae9cd996ca45532a98f81d817e`; API metadata reports 846,991,308 parameters. | Exceeds the full parameter target by over 4x; not downloaded. |
| Google ViViT-B/16x2 Kinetics-400 | Hub revision `google/vivit-b-16x2-kinetics400@8a7171a57f79b9aaa58bc8d977c002a0ea0f0d42`; 355,881,581-byte checkpoint (SHA-256 `160c79526e796d6101c4d9d0fe4000e26a6098a4d8e97f4cb8935fa6deb060d0`). The pinned Hub API identifies `google` as repository owner and exposes `MIT` in both `cardData.license` and the `license:mit` tag; the model card itself says Hugging Face authored it, not the ViViT team. The history shows checkpoint upload/fix commits by `jegormeister`, not Google. No model-team statement granting those exact weights was found. | Rights for these exact weights and parameter count were not independently verified; **REVIEW**, not an approved candidate. Not downloaded or loaded. |
| MCG-NJU VideoMAE-base | Hub revision `MCG-NJU/videomae-base@dc740ceda42fce44faed2ea03c6d447db72f6af9`; the Hub card labels the weights CC BY-NC 4.0 and reports 94.2M parameters / 376,873,760-byte safetensors. The model card says it was written by Hugging Face, not the model team. | Excluded from the commercial-use path due to the noncommercial license; not downloaded or loaded. |
| VATT | Official repository at `google-research/google-research@5b381db4139c3cbf25cc9a698c5fe9754215fae2` lists several Google Storage pretrained checkpoints, but the tested link redirected to Google sign-in; no exact weight license, hash or runtime was established. | Release listed, actual access and rights unresolved; legacy documented TensorFlow 2.7 / CUDA 10.1 stack is not the current local runtime. |
| MJEPA | Official `facebookresearch/MJEPA@82f3472219f2f58672999ed654314b34ad590c07` contains MIT training/evaluation code, but its Releases page is empty and no checkpoint assets were found. Published configs target ViT-L/ViT-g scale. | Code only; no pretrained weights available for a local audit. |
| ViT-Lens | Upstream repository says models were released; pinned HF ViT-Lens-L files total 12,326,294,229 bytes and pair modality weights with original CLIP ViT-L/14. | Pretrained and accessible but far beyond product size, not compatible with V-JEPA, and no local matched Decision evaluation. |

## Supplementary video-model screen (2026-10-11)

| Candidate | Pinned metadata | Disposition |
|---|---|---|
| Meta V-JEPA 2 ViT-G FPC64 | `facebook/vjepa2-vitg-fpc64-256@875c192b7b704b87d1e1d99345769632dd5f739a`; model card metadata declares Apache-2.0. Hub safetensors metadata reports 1,034,555,264 parameters and 4,138,311,608 bytes (SHA-256 `f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b`). Config: 40 layers, hidden size 1,408, 22 heads, 64 frames. The repository also contains a 16,464,208,210-byte original `.pth`; the pinned Hub card reports a 20.6GB repository. | The explicit Apache-2.0 model-card grant is useful, but this is V-JEPA 2 ViT-G, not the V-JEPA 2.1 ViT-B used by the Path A probes. Its >1B parameters, >4.1GB inference checkpoint, and 64-frame input make it far outside the 100–200M / <=200MB product target. Not downloaded or loaded; no teacher-distillation benefit has been measured. |
| OpenGVLab VideoMamba K400/F64 | Hub revision `OpenGVLab/VideoMamba@69502a732f8f69709cec9ed6c54d5345259fd825`; `videomamba_t16_k400_f64_res224.pth` is 28,327,498 bytes (SHA-256 `1323c8d2e23ac806e28b1abf4f37458db8b490b376987130ab1b6cc219bc6c16`). Hub API metadata advertises Apache-2.0, while the model card body says “Non-commercial license”; the upstream repository `LICENSE` at `37355c26d0ae99ca2459f6d4044a5f509031a79f` is Apache-2.0. The upstream model zoo describes K400 masked pretraining and further WebVid/multimodal-data pretraining for some variants. | License scope for this exact checkpoint is contradictory, so retain **REVIEW** and do not download or load it. Ask the model maintainers for the exact checkpoint and training-data terms before reconsidering. |
| Google VideoPrism-B | `google/videoprism-base-f16r288@be719a406d563b66f0ac969e7c94bab8e997c81a`; official source `google-deepmind/videoprism@c6eba0ece3dd49a24ffc44582966997a4a59f1d8`; 114,365,184 loaded parameters; 457,475,296-byte checkpoint (SHA-256 `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`). Official Hub card says Apache-2.0; official upstream says software Apache-2.0 and other materials, including released model checkpoints, CC-BY-4.0. Both grants permit use with attribution; preserve notices. | A matched 200-scene CLEVRER development comparison gave VideoPrism slightly better point estimates than V-JEPA in Accuracy/NLL/Brier, and a lower paired-bootstrap NLL interval. The split had already been used to select V-JEPA and selected VideoPrism's readout, so treat this as suggestive, not independent evidence. The readout overfit by epoch 24. Retain research-only pending independent scenes; not product-selected. FP32 checkpoint exceeds 200MB aspiration before compression. See `R2_PATH_A_PROBES.md`. |

## References

- [Official V-JEPA 2 repository and model release](https://github.com/facebookresearch/vjepa2), [pinned B/16 factory](https://github.com/facebookresearch/vjepa2/blob/main/src/hub/backbones.py), [upstream MIT license](https://github.com/facebookresearch/vjepa2/blob/main/LICENSE)
- [Ruri v3-30m model card](https://huggingface.co/cl-nagoya/ruri-v3-30m)
- [AST AudioSet model card](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593), [original AST repository](https://github.com/YuanGongND/ast)
- [ViT-Lens model zoo](https://github.com/TencentARC/ViT-Lens/blob/main/MODEL_ZOO.md)
- [Meta PE official implementation and checkpoint licenses](https://github.com/facebookresearch/perception_models/blob/main/apps/pe/README.md)
- [Google ViViT-B/16x2 pinned Hub card](https://huggingface.co/google/vivit-b-16x2-kinetics400/tree/8a7171a57f79b9aaa58bc8d977c002a0ea0f0d42), [pinned Hub API metadata](https://huggingface.co/api/models/google/vivit-b-16x2-kinetics400?revision=8a7171a57f79b9aaa58bc8d977c002a0ea0f0d42), [Hub commit history](https://huggingface.co/api/models/google/vivit-b-16x2-kinetics400/commits/main), [MCG-NJU VideoMAE-base Hub card](https://huggingface.co/MCG-NJU/videomae-base/tree/dc740ceda42fce44faed2ea03c6d447db72f6af9)
- [Meta V-JEPA 2 ViT-G pinned Hub card](https://huggingface.co/facebook/vjepa2-vitg-fpc64-256/tree/875c192b7b704b87d1e1d99345769632dd5f739a), [OpenGVLab VideoMamba pinned Hub card](https://huggingface.co/OpenGVLab/VideoMamba/tree/69502a732f8f69709cec9ed6c54d5345259fd825), [upstream model zoo](https://github.com/OpenGVLab/VideoMamba/blob/37355c26d0ae99ca2459f6d4044a5f509031a79f/videomamba/video_mm/MODEL_ZOO.md), [upstream license](https://github.com/OpenGVLab/VideoMamba/blob/37355c26d0ae99ca2459f6d4044a5f509031a79f/LICENSE)
- [Google VideoPrism-B pinned Hub card](https://huggingface.co/google/videoprism-base-f16r288/tree/be719a406d563b66f0ac969e7c94bab8e997c81a), [official README](https://github.com/google-deepmind/videoprism/blob/c6eba0ece3dd49a24ffc44582966997a4a59f1d8/README.md), [official license](https://github.com/google-deepmind/videoprism/blob/c6eba0ece3dd49a24ffc44582966997a4a59f1d8/LICENSE)

See [`R1_AUDIT.md`](R1_AUDIT.md) for local commands, hardware, output tensors and limitations.

## Text candidate update (2026-10-12)

The earlier table is a dated load audit. Subsequent task measurements supersede
its "quality unmeasured" entries; they do not change pinned source rights.

| Candidate | Loaded parameters | Evidence | Current disposition |
|---|---:|---|---|
| paraphrase-MiniLM-L3-v2, same pinned revision above | 17,389,824 plus196,865 readout | Criteria-preserving synthetic English validation Accuracy50.91%, NLL1.01490 | Frozen reference, not general conditional capability |
| cross-encoder/ms-marco-MiniLM-L6-v2@233902d25c440f23af6f7d6e94d2946bac0bee0a | 22,713,601 including classifier | Same validation: zero-shot45.98%, selected three-epoch final-block adaptation58.11%;1,774,849 existing parameters updated | Research candidate; not product-qualified |

The L6 base is Apache-2.0 per pinned card. Base SHA-256 is
`821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`;
90,870,598 serialized weight bytes. The selected delta is7,105,634bytes,
replaces existing tensors and adds no parameters. Its SHA-256 is
`5fa8323a47e72c3796d0da35618b603fa49784cacb670dee0e910e8d5757ed78`.
See R1_TEXT_CROSSENCODER_20261011.json and
R2_TEXT_CROSSENCODER_ADAPT_20261011.md for exact accounting and reload evidence.

The selected L6 failed the fixed64-state/256-question numerical condition probe:
all predictions B,35.94% Accuracy,0/128 lowest/highest answer changes. Candidate
serialization differs from training and remains an unresolved confound. Neither
this failure nor previous gains demonstrate a capacity ceiling; integration as a
general Decision scorer is not justified yet. See R2_TEXT_CONDITION_PROBE_20261012.md.

Per the user's clarification,100–200M is the underlying total parameter target;
<=200MB applies to measured quantized deployment packaging. Historical FP32
sizes above are reference measurements, not grounds alone to reject capacity.
No quantized quality/package result is available for the adapted L6 candidate.
