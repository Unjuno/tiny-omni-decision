# Frozen VideoSAUR train-only Decision probe — protocol 2026-10-13

Purpose: move from oracle mask diagnostics to actual conditional Decision
learnability, without spending on full validation extraction before the
representation can fit the existing train task. No claim of generalization from
this stage. The prior eight-video masks are not used as labels or model inputs.

## Fixed extraction now running

- All99 existing training clips/198 paired initial-final questions. Pair SHA
  8869bb903591af3a0995eeca0eeae1b50b990c8b08752a6e5d4d549f8a356fd7.
- Existing train media manifest SHA
  469299ad240ff85c47ba44abb8920aae9994b0e3ec15a3fca1373c49b2fa6ee3.
- All48,094 consecutive frames before each exclusive reveal cut; full clip for
  cut=-1. No clipping to a shorter budget or dropped failures. Every media hash
  verified before extraction and video IDs matched to the complete train corpus.
- Same frozen reconstructed93,530,201-parameter model, checkpoint, preprocessing
  and GPU seed17 policy as runtime/transfer reports. No parameter changes.
- Preserve predicted-state recurrence through encoder batches of6, initializing
  once per video. Seven128-dimensional posterior slots saved at32 uniform times
  including first/last frame; sampling does not skip recurrent processing.
- Append normalized time to each slot:224x129features per video. Do not assume
  slot identity is stable or use oracle mask assignments. No box annotations.
- Save media/prepared-input/feature hashes, exact frame IDs, source/config/helper
  hashes, count, timings and peak VRAM. Complete99-file ledger required before fit.
- New output C:/CodexArtifacts/pretrained-reuse-r1/videosaur-train-slots-01/.
  Source harness videosaur-source-audit-01/extract_train_slots.py and runtime
  loader hashed in protocol. Existing eight-clip masks cannot supply the missing
  slot vectors; this is a new feature extraction, not a repeat quality evaluation.

Observed eight-video speed suggested about1.26hours; this is a projection, not
completion time. Actual startup preflight passed48,094frames and the first459-frame
clip completed42.56seconds. Extraction was confirmed live via exec session10606.
Do not launch a duplicate based on this document: inspect the live handle and
output result/progress first. No readout fitting or validation extraction yet.

## Fixed next train-only fit

Use existing candidate-conditioned attention readout with input_dim129, hidden64;
MiniLM frozen candidate features unchanged (train file only). Default512dimension
and state layout remain compatible for legacy trajectory models. New head115,008
parameters, visual+text+head subtotal111,035,033; this omits other Omni components.

Seed17; AdamW lr.001,weight_decay.01; batches8 video pairs;50epochs. CPU only.
No best-checkpoint selection from train scores; save epoch0/final and every epoch's
train metrics. Keep all198questions, paired condition counts and53 changed-target
pairs. Record Accuracy/NLL/Brier/ECE, both-correct and condition switching, plus
cyclic mismatch/zero-video controls, gradient finiteness and exact final reload.
No hyperparameter sweep or extra seed. Fixed final endpoint is a diagnostic, not
a product candidate. Train fitting cannot establish overfitting absence or
validation performance; no train/validation gap is available at this stage.

If train condition sensitivity fails, do not automatically pay for257validation
video extractions. If it improves, specify the next fixed validation comparison
before running it; all prior development exposure must remain disclosed. No
sealed audit, model promotion, compression or shared-Lens experiment is authorized
by a low train loss. Checkpoint license and original non-shape config limitations
remain those recorded in R1. Existing artifacts are unchanged.

## Runner implementation follow-up

`scripts/train_slot_readout.py` implements the fixed train-only fit. It rejects
incomplete extraction, changed input hashes, wrong media/frame counts, altered
time values and mismatched question routing. It reads only the train feature
entry from the existing text ledger, never the validation feature file.

The unit smoke uses two synthetic video pairs in a separate pytest process:
50updates/50epochs, finite gradients, complete history and exact final reload
passed. Those synthetic metrics are not a real experiment result. No real-data
readout fitting has run yet.685tests passed,1skipped;Ruff passed.

After result.json confirms all99features, the intended command is:

```powershell
python -m scripts.train_slot_readout --features C:/CodexArtifacts/pretrained-reuse-r1/videosaur-train-slots-01 --output C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-slot-train-readout-01
```

This is a recorded command, not a scheduled or completed run. Completion is
required before executing it; preserve any existing output directory. Outputs
include ordered example identities/targets/options, all epoch metrics, initial
and final weights, final control predictions, hashes and reload verification.
No best.pt is produced and train results cannot select a deployment checkpoint.
