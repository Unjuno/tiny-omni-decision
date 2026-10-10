# R1 checkpoint feasibility audit

Date: 2026-10-10. Scope: pinned sources, rights metadata, safe checkpoint loading, actual local forwards, tensor shapes, parameter counts, runtime and memory. No training, fine-tuning, dataset tuning or sealed-audit access occurred.

## V-JEPA 2.1 ViT-B

- Upstream source: facebookresearch/vjepa2, commit 204698b45b3712590f06245fbfba32d3be539812.
- Checkpoint URL: https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt; 1,664,223,428 bytes; SHA-256 848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d; HTTP ETag be0dc26f052ae6a7476714cd53176836-199.
- torch.load(weights_only=True, mmap=True) found top-level keys batch_size, ema_encoder, encoder, epoch, loss, lr, opt, predictor, scaler, world_size; epoch 40. The EMA encoder has 158 float32 tensors, 86,833,152 parameters and 347,332,608 bytes. Predictor tensors are training-only for this path.
- Strict-loaded the upstream B/16 factory using a local injection of the pinned checkpoint. Separately strict-loaded ema_encoder into the pinned ViT-B constructor. This isolates the upstream factory test-only http://localhost:8300 URL from model/data directories without editing upstream files.
- CUDA image output: [1, 576, 768]; 8-frame video output: [1, 2304, 768]; all finite. Factory and direct-EMA outputs had max absolute difference 0 for both inputs.
- Input was an existing plumbing image and a 2-frame plumbing video. To form 8 frames the deterministic sampler repeated the two frames as [0,0,0,0,1,1,1,1]. This tests loading, tensor dimensions and preprocessing only; it says nothing about temporal task quality.
- Both official and direct reference model were resident during parity comparison. Peak allocated/reserved GPU memory was 831,897,600 / 985,661,440 bytes. Primary forward time was 0.297s for the image and 0.191s for video on this run; process timing is not a stable benchmark.
- License evidence was rechecked against the exact pinned source commit. It includes both `LICENSE` (MIT) and `APACHE-LICENSE`; its README says most of V-JEPA 2 is MIT and names three source files under Apache-2.0, while linking the checkpoint in its model table. It does not explicitly identify the externally hosted `vjepa2_1_vitb_dist_vitG_384.pt` weight file's license or say that either repository license covers that download. Keep checkpoint rights status **REVIEW**; do not redistribute or package the weight for product use until the grant is explicit. This uncertainty does not change the verified tensor/load results. Candidate status is also recorded in `R1_CANDIDATE_INVENTORY.json`.

### Rights follow-up, 2026-10-11

An additional primary-source check found that Meta's June 2025 V-JEPA 2 announcement says Meta is making V-JEPA 2 code and model checkpoints available for commercial and research applications. That announcement predates the pinned V-JEPA 2.1 release and does not name the exact 2026 V-JEPA 2.1 ViT-B file. The pinned repository's README links the 2.1 checkpoint and says the majority of V-JEPA 2 is MIT-licensed, but the README's license section does not explicitly state that the separately hosted 2.1 weights are covered. Therefore the official announcement is useful context but does not close the exact-weight grant gap; keep status **REVIEW** until a license statement that clearly applies to that checkpoint is found. This is a provenance/legal-status update only; no checkpoint was re-downloaded or modified.

Sources: [pinned V-JEPA 2.1 repository README](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/README.md), [Meta's June 2025 V-JEPA 2 announcement](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/), and [Meta's V-JEPA project page](https://ai.meta.com/vjepa/).

Command (report saved outside Git at C:\CodexArtifacts\pretrained-reuse-r1\vjepa21-audit.json):

    $env:PYTHONPATH='C:\CodexArtifacts\pretrained-reuse-r1\python-deps'
    py -3.11 scripts\audit_vjepa21_checkpoint.py --source-root C:\CodexArtifacts\pretrained-reuse-r1\vjepa2-source --checkpoint C:\CodexArtifacts\pretrained-reuse-r1\vjepa2_1_vitb_dist_vitG_384.pt --image C:\CodexArtifacts\eg2-media-smoke-7443c7ut\image.png --video C:\CodexArtifacts\eg2-media-smoke-7443c7ut\video.mp4 --output C:\CodexArtifacts\pretrained-reuse-r1\vjepa21-audit.json --expected-checkpoint-sha256 848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d --device cuda --video-frames 8

## Japanese text and environmental-audio candidate forwards

- Text: Ruri v3-30m, revision 24899e5de370b56d179604a007c0d727bf144504, Apache-2.0. 36,705,536 float32 parameters; file 146,828,152 bytes; SHA-256 e94155e342cbfc33280c345c6e2912905fb5de6e1a3adeb938f395aafd2f8832. A Japanese and an English string tokenized together and produced finite hidden states [2,32,256] and pooled vectors [2,256]. The model card lists Japanese as the supported language, so English has no quality claim.
- Audio: AST AudioSet, revision f826b80d28226b62986cc218e5cec390b1096902, BSD-3-Clause. 86,594,063 float32 parameters; file 346,404,948 bytes; SHA-256 ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d. A 3,200-sample / 16 kHz fixture passed through the feature-extractor contract and returned finite logits [1,527]. The smoke WAV is a plumbing fixture, not a Speech Commands example.
- Transformers 5.6.2, PyTorch 2.6.0+cu124, CUDA 12.4. Sequential text and AST forward times were 0.201s and 0.102s respectively in this single run; model load times were 0.794s and 0.231s. Both models were resident during the combined pass; peak allocated/reserved memory 554,551,296 / 603,979,776 bytes.
- The first AST audit used the Transformers fallback filterbank and warned about zero-valued mel filters. Installed matching `torchaudio==2.6.0+cu124` into the isolated audit dependency target only, then reran through the official Kaldi fbank path. The warning disappeared; global packages were not changed. Latest output is `C:\CodexArtifacts\pretrained-reuse-r1\text-audio-audit-v3.json` (outside Git).

Command:

    py -3.11 scripts\audit_hf_pretrained_candidates.py --text-model C:\CodexArtifacts\pretrained-reuse-r1\ruri-v3-30m --text-revision 24899e5de370b56d179604a007c0d727bf144504 --audio-model C:\CodexArtifacts\pretrained-reuse-r1\ast-audioset --audio-revision f826b80d28226b62986cc218e5cec390b1096902 --audio C:\CodexArtifacts\eg2-media-smoke-7443c7ut\audio.wav --output C:\CodexArtifacts\pretrained-reuse-r1\text-audio-audit-v3.json --device cuda

## English text and speech candidate forwards

- English text: `sentence-transformers/paraphrase-MiniLM-L3-v2@4ca70771034acceecb2e72475f72050fcdde4ddc`, Apache-2.0. Locally loaded 17,389,824 float32 parameters; `model.safetensors` is 69,569,488 bytes, SHA-256 `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`. Two English strings produced finite mean-pooled vectors of shape [2,384]. Load time was 0.190s on the local CUDA device.
- Speech: `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af`, model card Apache-2.0. Locally loaded 37,760,640 parameters; `model.safetensors` is 151,061,672 bytes, SHA-256 `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`. The frozen encoder returned finite [1,1500,384] features from 1 second of zero-valued 16 kHz audio. This is a tensor plumbing check only, not speech quality. Load time was 0.247s and encoder forward 0.083s.
- These four proposed towers (V-JEPA, Ruri, MiniLM, Whisper) total 178,689,152 parameters and 714,791,920 FP32 weight bytes. This is inside the 200M parameter target but far above the 200MB package target. Keeping both text towers covers the audited Japanese and English scopes; neither alone proves bilingual task quality. Replacing AST with Whisper for spoken-keyword tasks is a task-fit hypothesis, not an evaluation result.
- These outputs were produced in a separate read-only process with `PYTHONPATH=C:\CodexArtifacts\pretrained-reuse-r1\python-deps`; no model parameters were updated. The R2 report and dataset results are separate.

## R1 decision

Technical feasibility: PASS for local V-JEPA image/video, Ruri Japanese text, MiniLM English text, AST environmental audio and Whisper speech encoders. The V-JEPA checkpoint rights gate remains REVIEW, so R1's legally cleared video-candidate exit is not yet satisfied for product use. Deployment fit: parameter target PASS for the four-tower proposal (178,689,152), package-byte target FAIL before compression (714,791,920 bytes), runtime and task quality UNCERTAIN. Typed Decisions Synth and CLEVR-4 are now used for synthetic readout probes only; neither closes real-world quality gates. Open-Jev is inaccessible anonymously (HTTP 401 on 2026-10-10); its pinned manifest remains unchanged.

The only local Python processes found were Django development servers, not training. The RTX 3080 Laptop GPU was idle (11 MiB used, 0% utilization) before the read-only forward audits. No legacy run or worktree was changed.

## V-JEPA 2.1 checkpoint-rights follow-up (2026-10-11)

The official Meta V-JEPA page links the public V-JEPA 2 download and describes
release for community use, while the upstream repository README identifies
V-JEPA 2.1 and links its ViT-B/16 checkpoint. The repository exposes MIT and
Apache-2.0 license files for code. In the reviewed materials, no separate
grant is attached to the exact `vjepa2_1_vitb_dist_vitG_384.pt` weight file;
the community-release statement alone does not specify downstream commercial
rights, redistribution, or derivative-weight terms. Therefore the existing
checkpoint status stays **REVIEW**, and its established local forward results
remain research-only. This follow-up does not change or overwrite the prior
audit or candidate inventory.

Sources inspected 2026-10-11: [official Meta V-JEPA release page](https://ai.meta.com/research/vjepa/), [upstream README at the pinned source commit](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/README.md), [upstream MIT license](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/LICENSE), and [upstream Apache license](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/APACHE-LICENSE). The public Hugging Face V-JEPA 2 collection was also checked; it does not list the pinned 2.1 ViT-B/16 checkpoint as a model-card alternative.

## Additional compact video candidates (2026-10-11)

Two smaller released video encoders were checked without downloading weights.
The exact Hugging Face revisions, file hashes/sizes and conservative rights
dispositions are in `R1_CANDIDATE_INVENTORY.json`.

- `google/vivit-b-16x2-kinetics400@8a7171a57f79b9aaa58bc8d977c002a0ea0f0d42`
  lists a 355,881,581-byte checkpoint. Its card frontmatter says MIT, but the
  same card says Hugging Face authored it rather than the model team; the Hub
  API's top-level `license` field is null, and the checkpoint-add commit is
  attributed to a non-Google account. I did not find an independent grant for
  these exact weights or verify the exact parameter count. Keep it **REVIEW**;
  it was not downloaded or loaded.
- `MCG-NJU/videomae-base@dc740ceda42fce44faed2ea03c6d447db72f6af9` lists
  94.2M parameters and a 376,873,760-byte safetensors file, but its Hub card
  declares CC BY-NC 4.0. Exclude it from the project's commercial-use path;
  it was not downloaded or loaded.

These screens do not resolve the exact V-JEPA 2.1 checkpoint grant and do not
close R1's legally-cleared video-candidate exit. Sources: [pinned ViViT Hub
card](https://huggingface.co/google/vivit-b-16x2-kinetics400/tree/8a7171a57f79b9aaa58bc8d977c002a0ea0f0d42), [pinned ViViT Hub API
metadata](https://huggingface.co/api/models/google/vivit-b-16x2-kinetics400?blobs=true), and [pinned VideoMAE
card](https://huggingface.co/MCG-NJU/videomae-base/tree/dc740ceda42fce44faed2ea03c6d447db72f6af9).

## Supplementary video-model screen (2026-10-11)

Two further public video checkpoints were screened by their pinned Hub metadata;
no model weights were downloaded or loaded.

- `facebook/vjepa2-vitg-fpc64-256@875c192b7b704b87d1e1d99345769632dd5f739a`
  is explicitly tagged Apache-2.0 on Meta's Hub model card. The API reports
  1,034,555,264 parameters and a 4,138,311,608-byte safetensors file
  (SHA-256 `f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b`).
  Its pinned config specifies 40 layers, hidden size 1,408, 22 attention heads,
  and 64 frames; the repository totals 20.6GB including the 16,464,208,210-byte
  original checkpoint. This is the V-JEPA 2 ViT-G model, not the V-JEPA 2.1
  ViT-B Path A encoder. It is a possible large teacher only; its size and input
  budget are incompatible with the compact product target, and no benefit from
  distilling it has been measured.
- `OpenGVLab/VideoMamba@69502a732f8f69709cec9ed6c54d5345259fd825` exposes a
  K400 / 64-frame checkpoint of 28,327,498 bytes (SHA-256
  `1323c8d2e23ac806e28b1abf4f37458db8b490b376987130ab1b6cc219bc6c16`). Its
  Hub API metadata says Apache-2.0, but the same model card body says
  “Non-commercial license.” The upstream repository's root `LICENSE` is
  Apache-2.0; its model zoo describes masked pretraining on K400 and separately
  lists WebVid/multimodal-data variants. Because the exact checkpoint grant and
  training-data scope are ambiguous, keep it **REVIEW** and do not use it in a
  product or evaluation run until the model maintainers clarify.

Sources: [Meta's pinned V-JEPA 2 ViT-G card](https://huggingface.co/facebook/vjepa2-vitg-fpc64-256/tree/875c192b7b704b87d1e1d99345769632dd5f739a), [pinned model README/config](https://huggingface.co/facebook/vjepa2-vitg-fpc64-256/blob/875c192b7b704b87d1e1d99345769632dd5f739a/README.md), [OpenGVLab's pinned VideoMamba card](https://huggingface.co/OpenGVLab/VideoMamba/tree/69502a732f8f69709cec9ed6c54d5345259fd825), [upstream model zoo](https://github.com/OpenGVLab/VideoMamba/blob/37355c26d0ae99ca2459f6d4044a5f509031a79f/videomamba/video_mm/MODEL_ZOO.md), and [upstream license](https://github.com/OpenGVLab/VideoMamba/blob/37355c26d0ae99ca2459f6d4044a5f509031a79f/LICENSE).

## Google VideoPrism-B pinned checkpoint audit (2026-10-11)

This is a new official, compact-enough video candidate. Unlike V-JEPA 2.1,
its exact public checkpoint has an explicit rights basis: the official Google
Hub model card labels the checkpoint repository Apache-2.0, and the official
upstream README says software is Apache-2.0 and all other released materials
are CC-BY-4.0. The official upstream repository identifies the VideoPrism
weights as released checkpoints; CC-BY-4.0 permits commercial reuse with
attribution. Preserve both notices when redistributing. The official Hub card
and upstream repository use different license labels, but both are permissive
grants for this released weight. No inference is made from code license alone.

- Exact weights: `google/videoprism-base-f16r288@be719a406d563b66f0ac969e7c94bab8e997c81a`, file `flax_base_f16r288_repeated.npz`, 457,475,296 bytes, verified SHA-256 `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29` (matches the Hub LFS SHA-256).
- Official implementation: `google-deepmind/videoprism@c6eba0ece3dd49a24ffc44582966997a4a59f1d8`. Used `vp.get_model("videoprism_public_v1_base")` and `vp.load_pretrained_weights(..., checkpoint_path=<verified local file>)`; no implicit latest-checkpoint download. The loaded tree contained 114,365,184 float32 parameters (457,460,736 tensor bytes).
- A deterministic synthetic 8-frame `[1,8,288,288,3]` float32 input in `[0,1]` produced finite `[1,2048,768]` float32 token features. Output SHA-256: `df185755af2fabc6c8dcf0765743edd5cbb01cf6bb6bc7d968e469a929b9796f`.
- This is a load/shape/numerical smoke test only; task quality is unmeasured. Model-card benchmark numbers are not used as CLEVRER or product claims. It is 114.4M parameters, but the uncompressed checkpoint alone is 457.5MB, above the 200MB package aspiration; compression and complete runtime costs remain unmeasured.
- Environment: Windows x86-64, Python 3.11.16, JAX 0.6.2, Flax 0.11.1, TensorFlow CPU 2.18.1, NumPy 1.26.4; one JAX CPU device. Windows JAX GPU is unsupported by the official platform matrix. First 8-frame compile+forward took 7.91s; peak process working set was 3,917,281,280 bytes. The RTX 3080 Laptop GPU was idle and not used for this JAX run. A newer JAX/Flax attempt was blocked by the machine's Smart App Control policy on an unsigned `_sdy.pyd`; the isolated environment was pinned to the compatible 0.6.2/0.11.1 pair, which imported and ran without bypassing the policy. Existing Python environments and user processes were not changed.
- External run directory: `C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1`; result JSON SHA-256 `9a0521eedfc656751a24c2d9d639998c9e57dd53747fffaaf698acbd1a4815e`, external audit script SHA-256 `a1c3d210f84bd6f6df3fc90f9dcbea31529863e12839480203a96be200c0b526`, and environment freeze SHA-256 `91012917e8fe29d3a3bedfecc8e3fc641a7a055580bc8c5f4eea1457519c0aad`. The downloaded weights, source checkout, venv, and report remain outside Git.

The repository audit driver was subsequently run against the same pinned local
checkpoint/source in a separate, non-overwriting output file. It reproduced
the same parameter count, output shape, finite values, and feature SHA-256;
the recorded report is
`C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\audit-result-repro-v2.json`
(SHA-256
`0c259eaaa0521236e775d76f773ca992c7db757225e099aeb34917651fe3af25`).
The checked-in driver hash is
`0f57e97bc1e43ea2248e9537aa922a666ef62ad6db842b3ca291ceca1300dd19`.
This second compile-plus-forward took 6.13 seconds, with a 4,113,592,320-byte
peak working set; run start was recorded at `2026-10-10T20:22:24.014815+00:00`.

Sources: [official VideoPrism README at the pinned implementation commit](https://github.com/google-deepmind/videoprism/blob/c6eba0ece3dd49a24ffc44582966997a4a59f1d8/README.md), [official pinned license](https://github.com/google-deepmind/videoprism/blob/c6eba0ece3dd49a24ffc44582966997a4a59f1d8/LICENSE), and [pinned Google Hub model card](https://huggingface.co/google/videoprism-base-f16r288/tree/be719a406d563b66f0ac969e7c94bab8e997c81a). The current [JAX platform matrix](https://docs.jax.dev/en/latest/installation.html) lists Windows x86-64 CPU support and no native Windows NVIDIA GPU support.
