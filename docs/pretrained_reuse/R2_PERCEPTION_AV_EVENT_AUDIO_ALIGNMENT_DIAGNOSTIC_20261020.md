# AV event audio-alignment diagnostic — 2026-10-20

## Status and scope

Status: **completed**. This is a **post-hoc, fixed-checkpoint intervention diagnostic** on the same
previously unused Perception Test TRAIN video subset described in
[`R2_PERCEPTION_AV_EVENT_FRESH_HOLDOUT_PROTOCOL_20261020.md`](R2_PERCEPTION_AV_EVENT_FRESH_HOLDOUT_PROTOCOL_20261020.md).
It is not a preregistered independent quality evaluation, and the original
holdout results were already observed. This follow-up does not tune, calibrate,
select, or update a model. It can characterize sensitivity to audio
alignment, but cannot establish that the model is correct or that audio is
causally necessary. It must not be used to choose another checkpoint or
architecture.

## Fixed inputs

- Feature manifest SHA-256:
  `5a4fed0a0419fe1091364b5b240b371e47c64ffff8516a2fe179eaac998fd1a4`.
- Selected joint readout SHA-256:
  `be9a09f1c57c679b176d69c9a14825d4d38d053bee4b2ea5818ee108b7da8a36`.
- Matched video-only prediction SHA-256:
  `4acac4d853302a369aca1aabbe3368bf923a0a0ee41fcf50a4a8a74f3a745a71`.
- All 504 events and 200 video groups are retained with the original option
  lists, target indices, candidate order, and fixed selected checkpoint.

## Interventions and analysis

For every event, evaluate the fixed joint checkpoint with:

1. the original event audio and candidate visual features;
2. a zero audio feature vector, explicitly treated as an out-of-distribution
   ablation rather than a missing-modality token; and
3. audio taken from a different video group, using a deterministic SHA-256
   ordering of video IDs and a cyclic donor mapping. Videos are paired with the
   next video in that order; event IDs are sorted within each video, and target
   events map modulo the donor video's event count. Candidate visuals and
   targets remain fixed.

Report Accuracy, NLL, Brier, and fixed 10-bin ECE for each arm. Use 10,000
video-cluster bootstrap draws with seed 17 for paired aligned-minus-control
differences in Accuracy and NLL, plus aligned joint versus the previously
saved video-only baseline. No labels, options, candidate order, or checkpoints
are changed. Inference is CPU-only; no media is decoded, no encoder is loaded,
no optimizer is created, and no GPU or sealed/test data is used.

The cross-video assignment changes both audio content and alignment. A
performance drop therefore indicates sensitivity to that intervention; it is
not proof of valid cross-modal reasoning. The zero-vector arm is also a stress
ablation, not a realistic sensor dropout behavior. Findings remain conditional
on this nonblind TRAIN subset and unknown participant/source grouping.

## Results

All 504 events across 200 videos completed on CPU in 1.21 seconds of model
inference. No GPU, encoder, optimizer, test, or sealed-audit data was used.

| Condition | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Joint, original aligned audio | 0.3948 | 1.8466 | 0.7622 | 0.0742 |
| Joint, zero audio feature | 0.3631 | 1.9181 | 0.7822 | 0.0577 |
| Joint, audio from another video | 0.3214 | 2.0628 | 0.8250 | 0.1315 |
| Matched video-only reference | 0.3532 | 2.0045 | 0.8111 | 0.1415 |

Video-cluster paired bootstrap, 10,000 draws, seed 17 (aligned minus control):

| Contrast | Accuracy difference (95% interval) | NLL difference (95% interval) |
|---|---:|---:|
| Zero-audio intervention | +0.0317 [−0.0076, +0.0705] | −0.0715 [−0.1291, −0.0134] |
| Cross-video audio | +0.0734 [+0.0338, +0.1155] | −0.2162 [−0.2999, −0.1388] |
| Video-only reference | +0.0417 [−0.0078, +0.0903] | −0.1579 [−0.2478, −0.0670] |

Aligned audio has lower NLL than both interventions on this subset. Its
accuracy advantage over zero audio is uncertain; cross-video swapping reduces
Accuracy. This indicates sensitivity to audio content/alignment, but does not
establish that both modalities are necessary for correct decisions. The zero
vector is out-of-distribution, and cross-video audio may be semantically
incompatible with its candidate scene. This post-hoc diagnostic is not an
independent blind evaluation and must not be used for tuning or model
selection.

The external receipt is
`C:\\CodexArtifacts\\pretrained-reuse-r2\\perception-av-event-audio-alignment-diagnostic-01\\result.json`,
SHA-256 `5969838bef906f9183d5738b2c50c6e974302fcbf1b212e1ee6d474c69e50641`.
Prediction hashes are `ce1132d8cd9420931c94aeed9fd97ec6e0fe9e75da496071b1bd0af2fe2f8642`
(aligned), `86f1cfaa6a0a82d184c1974e83dd35fe7563ef5abcf254036706e280601a5a08`
(zero audio), and `b9125d66e714787b1971edbf321f8d68ff9adef174ef906b434f56cadd7b81d5`
(cross-video audio). The donor map is
`cross_video_audio_donor_map.json`, SHA-256
`c6b7c2370ccb9707c3fad044393d5b4148e53d157c7fa4d133406a7d68f80fe4`;
all mappings cross video groups.

Runner SHA-256: `7d1e9360a2e26c5cecc08cf81f9a663d4d708b911b3a3e6ab7ce0d65165c7711`.
Targeted test SHA-256: `9f84da0cfcc9ba142759b937661656bb2dfbad2a90385ff9846b99e1d813ace2`.

## Execution

The CPU-only runner is
`scripts/diagnose_perception_av_event_audio_alignment.py`. It refuses an
unexpected feature hash, checkpoint hash, or baseline-prediction hash and
wrote to a new output directory under `C:\\CodexArtifacts\\pretrained-reuse-r2\\`.
Runtime recorded for the diagnostic: Python 3.11.9 and PyTorch 2.6.0+cu124.
`CUDA_VISIBLE_DEVICES` was empty for the run; the receipt confirms `gpu_used: false`.
The result file was written at 2026-10-20 18:46:20 local time.

The command was:

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
python -m scripts.diagnose_perception_av_event_audio_alignment `
  --features C:\\CodexArtifacts\\pretrained-reuse-r2\\perception-av-event-fresh-holdout-features-01\\features.jsonl `
  --features-sha256 5a4fed0a0419fe1091364b5b240b371e47c64ffff8516a2fe179eaac998fd1a4 `
  --output C:\\CodexArtifacts\\pretrained-reuse-r2\\perception-av-event-audio-alignment-diagnostic-01
```

Existing holdout receipts and model artifacts remain immutable.
