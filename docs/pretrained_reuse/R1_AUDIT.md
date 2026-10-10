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
