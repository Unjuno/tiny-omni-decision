# Compact semantic detector: real CPU load and fresh repeat

The full native <=200M text/image/audio/video Omni calibrated Decision/JSON,
joint/OOD and95–98% goal remains open. This is one independently audited
**component**, not a selected product or a claim of useful object discovery.

## Why and released identity

Existing raw grid readouts overfit and TinyCLIP semantic cup ranking failed to
improve its baseline. Do not repeat those closed fits/prompts/seeds. A pretrained
semantic object detector is a distinct feature candidate that can be inspected
without changing the currently active frozen SAM2 memory comparison. No adoption
or follow-up quality experiment is justified by mechanics alone.

Released `PekingU/rtdetr_v2_r18vd@5650961749fa93567c0d46fc7f43ea4f9e914107`.
The [pinned official model card](https://huggingface.co/PekingU/rtdetr_v2_r18vd/blob/5650961749fa93567c0d46fc7f43ea4f9e914107/README.md)
declares Apache-2.0; [official upstream code](https://github.com/lyuwenyu/RT-DETR)
also declares Apache-2.0. Weights/card/model-config/processor-config are verified
against immutable LFS SHA256/Git blob pins, with metadata retained outside Git.
Published weight SHA256: `d18309d0d7ea57048138885c4c6ecfcb1e24506fc6153b94ad484f8ab62c7115`.
Original COCO training-media rights were not separately audited; none are ingested
here. The repository's R1 released-weight policy is ALLOW, not a training-source
rights conclusion. Rounded upstream detection size/AP/FPS are not used as project
parameter/Decision/latency evidence.

## Actual execution

[Protocol](R1_RTDETR_V2_R18_CPU_PROTOCOL_20261014.md) ·
[Machine evidence](R1_RTDETR_V2_R18_CPU_RESULTS_20261014.json).

```powershell
python -m scripts.audit_rtdetr_cpu_candidate --output C:/CodexArtifacts/pretrained-reuse-r1/rtdetr-v2-r18-audit-01/runtime-01
python -m scripts.audit_rtdetr_cpu_candidate --output C:/CodexArtifacts/pretrained-reuse-r1/rtdetr-v2-r18-audit-01/runtime-reload-01
python -m scripts.record_rtdetr_cpu_audit
```

Both processes hide CUDA before importing Torch; all parameters on CPU/FP32,
eval/frozen, seed17/deterministic/4threads. Existing Transformers5.6.2,
PyTorch2.6.0+cu124 and PyAV; no environment upgrade or cloud. Host CPU is
Intel i7-12700H (14cores/20logical processors), observed physical RAM34,087,497,728bytes.
Timings include concurrent frozen SAM2 GPU work and use4CPUthreads.
Native640×640 processor and all300queries/80classes, no NMS/threshold/class filter. The immutable
config uses an older class-name capitalization; use the actually installed native
`RTDetrV2ForObjectDetection`. CPU custom GPU kernels explicitly disabled. Parse
the released card's UTF-8 frontmatter rather than Windows CP932. Both issues were
identified before actual runtime; no model/version/data fallback.

One source image: raw frame0 of the already-admitted known TRAIN video10103;
media SHAe9914a811679e99f26246ab0c1f89c582914e37433189bf925f06f5793ca38ed.
No source annotations, targets, actor splits, official validation or sealed set.
This is repeated mechanics on one observed image, not a held-out detection test.

| Measurement | First CPU process | Fresh CPU model process |
|---|---:|---:|
| Body seconds (includes imports/load) |15.1311|15.2486|
| First forward seconds |0.565843|0.590734|
| Second forward seconds |0.506425|0.544424|
| Peak process bytes |892,301,312|892,424,192|
| CUDA initialized |false|false|

Actual unique registered parameters **20,174,608** /80,698,432FP32bytes.
Registered buffers35,154elements; total released tensor inventory20,209,762,
with80,839,232tensor payload bytes. Four released files total80,914,063bytes;
model.safetensors80,904,640bytes. These are standalone component sizes, not an
all-resident Omni budget or quantized package.

All arrays retained: input `[1,3,640,640]`, logits `[1,300,80]`, boxes `[1,300,4]`.
Producer NPZ save/readback is shape/dtype/byte-identical. Separate fresh model load
matches every processed-input/logit/box byte and before/after persistent-state hash
`13d6b3e50358979e070e9ee5d77f3a84ec13172377232cb7e3d03f33cb7bb661`.
State scope is parameters and persistent buffers; no gradients or CPU RNG changes.
The recorder independently rechecks all release/input/library/source/output pins,
native array inventories/finite/bbox bounds and fresh outputs. It is not independent
model-implementation or upstream conversion parity verification.

## Publication and unfinished gates

Exact execution-source snapshots and hashes are in both artifact directories.
Source base commit `b0d547e3f5edcc90b93e244a986ada3e69e666a9`, implementation
**uncommitted during execution** to preserve the active SAM2/fresh source commit.
The small source/tests/report are published together after that experiment's
disk-reserve interruption; the original execution provenance stays uncommitted.
Four metadata regression tests pass; model weights/NPZ/media remain outside Git.

No Accuracy/NLL/Brier/ECE, semantic cue/cover/identity quality, video temporal
quality, action readout, joint fusion, OOD, package integration or promotion.
The SAM2 comparison is incomplete (see the separate interruption report).
Complete its admission and define a distinct fixed semantic input test
before any task-quality run. No further current readout fitting, quantization or
legacy changes follow from this CPU result.
