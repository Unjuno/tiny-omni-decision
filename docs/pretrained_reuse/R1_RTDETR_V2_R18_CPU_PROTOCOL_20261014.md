# RT-DETRv2 R18: bounded independent CPU candidate audit

The active SAM2 full mask-memory inference stays untouched; no concurrent GPU
test or training. R0–R8/full native <=200M Omni conditional/joint/OOD95–98% goal
remains open. Existing semantic TinyCLIP ranking and grid readout fits are closed.
This independently checks a compact pretrained **semantic object detector**, not
another fit, prompt/threshold/seed search or a replacement for the active experiment.

Pinned model `PekingU/rtdetr_v2_r18vd@5650961749fa93567c0d46fc7f43ea4f9e914107`,
Hub Apache-2.0 card, public/ungated, published80,904,640-byte weight LFS SHA256
`d18309d0d7ea57048138885c4c6ecfcb1e24506fc6153b94ad484f8ab62c7115`.
Official model release: https://huggingface.co/PekingU/rtdetr_v2_r18vd
Upstream code/license: https://github.com/lyuwenyu/RT-DETR
Do not treat COCO detection AP or rounded20M as project Decision Accuracy or
measured all-resident Omni size. Record actual checkpoint tensor inventory,
unique registered parameters/buffers, file sizes/SHA and installed source hashes.

Only selected README/config/processor/safetensors files in a new output. No
new packages/runtime upgrade, unsafe pickle, remote model code, cloud or unknown
media. Existing Transformers5.6.2/PyTorch2.6.0+cu124/PyAV environment. Hide GPUs in
this independent process before importing Torch; assert CPU tensors and no CUDA
initialization. FP32/eval/frozen/seed17/deterministic/4threads, existing processor
settings, disable custom GPU kernels. CPU body120s, process3GiB, disk8GiB reserve.

One fixed real image: source frame0 of already-admitted TRAIN video10103, immutable
media SHAe9914a811679e99f26246ab0c1f89c582914e37433189bf925f06f5793ca38ed.
Validate existing train-manifest pin469299ad240ff85c47ba44abb8920aae9994b0e3ec15a3fca1373c49b2fa6ee3.
No labels, options, category filter, NMS, threshold, source role or QA scoring.
Native preprocess and forward twice on that exact image. Retain **all300queries**
and all80 configured raw class logits/boxes, save/load byte comparison, state/flags/gradients/
CPU RNG check. This is load/mechanics, not semantic discovery or temporal evidence.
All four downloaded files are checked against the immutable revision metadata:
weight LFS SHA256 and actual Git blob SHA1 for card/model-config/processor-config.
Record SHA256 only after these release pins pass. Parse card frontmatter as UTF-8;
Windows CP932 is not the card encoding. Model config stores the historical
`RtDetrV2ForObjectDetection` name; existing Transformers5.6.2 exports the native
`RTDetrV2ForObjectDetection`, used explicitly without remote code or architecture
substitution. Retain pinned API file metadata in the new artifact directory.

The code is new/uncommitted while SAM2's source commit is fixed. Save exact source
snapshot/hash alongside the CPU result and record base commit; do not present
uncommitted code as committed. Publish after main/fresh finishes, together with
normal tests/manifest/CI checks. No old code/config/artifact overwrite or model
promotion. Failure is preserved without silent config/model/version fallback.
