# HoloAssist window-label alignment audit — 2026-10-20

## Finding

This post-hoc CPU audit reproduced the exact source-label to action-event to
30-second-window mapping for all selected HoloAssist training and validation
data, then stratified the already-saved validation predictions by whether an
annotated instructor correction cue occurred inside the model's input window.
It did not load model weights, decode media, train, or open test/sealed data.

The cue is not an accidental split leak: it is valid contemporaneous audio
under the current task definition. It is, however, directly correlated with
the target “did any action in this window contain an incorrect action?” and
therefore makes the task unsuitable as evidence that audio/video fusion is
necessary for autonomous error recognition.

## Pinned inputs and integrity

- Source implementation and result context: immutable commit
  [c36ce36](https://github.com/Unjuno/tiny-omni-decision/tree/c36ce369e8687205fcc2d7670ab3e12a3b226fd3).
  The fixed training run itself records source commit
  327d63e61b279e293520e7afaef3f22294ff214b and config SHA-256
  f4de89195cbe5928a65a7b0ca634ee86ea3e25548856db07fbeca020d307ac32.
- Train/validation-only source annotation SHA-256:
  cc7898b49958a62fe021ae2ffa53a709c3fd6f45fd3f893960b8aac6d13dfe9c.
- Fixed selection SHA-256:
  577adbafe6be869de3aff3b814acf91db7be582a42e0a7a83e7ee773dde0e8b7.
- Action-event JSONL hashes (train / validation):
  bc4e34f87a8b4b3c8e493f3d1c60b90ed1ea8cf4f5438b0b8d263313c4f319b1 /
  1166c84b23720aa6239d085cad203c56224a0917bb3d8712b4448f5fd5d9555a.
- Window JSONL hashes (train / validation):
  52e028767aa00c5d2effc2735a1a404d1a18a64a7ee1a39d947e1c59f7041b81 /
  b58c198d611ceab8d2d435004203ef8f5079a389e3560a41937118480c64c671.
- Readout training metadata SHA-256:
  a96d4a873da134b02ed9269316360161f2a96a9b678750593607a60f913fff2f.
  The audit enforces its completed status, training source commit, config hash,
  and sealed-audit-not-loaded field.
- Audio/video/joint selected checkpoint hashes and exact prediction hashes are
  enforced by the audit code and recorded in the receipt.
- Accepted external audit receipt:
  C:\CodexArtifacts\pretrained-reuse-r2\holoassist-label-window-audit-05\result.json,
  SHA-256
  3a3f9e48a3a7a82c98ecd5dfc0c7269f6b26247359dccea8e3e89cb5005a66ab.
  Audit script SHA-256:
  73ec91ffbab4eb64a3f6a80425f904e496c6db6d2acb9a1c4a942b0e2fffb18a.
  Tests SHA-256:
  5663b0b2ebd1d5cd9e66e90a4c98ad0af60f88bd92943419260a6175cd759c64.

Attempt directories are immutable. Attempt 01 used an overly strict single-ID
conversation join and undercounted linked speech; it is superseded. Attempts 02
and 03 corrected comma-separated sequence links and separated explicit
correction-purpose speech. Attempt 04 pinned prediction/checkpoint files; attempt
05 additionally verifies the exact completed run metadata and is the accepted
receipt.

## Audit method

The runner verifies all pinned file hashes before processing. It joins each
retained fine-action event back to its train/validation source annotation,
checks correctness subtype and exact start/end times, independently assigns
each event to exactly one window by its end time, and reconstructs every
window's event IDs, target, loss mask, and action counts. It resolves the
annotation's Incorrect Action Corrected by references and comma-separated
conversation sequence IDs. A correction cue counts only when the linked
conversation overlaps the same 30-second input window. An explicit correction
cue additionally requires the source conversation purpose to say it corrects a
wrong action.

All selected training and validation windows passed those integrity checks:

| Split | Sessions | Fine-action events | Labeled windows | Positive windows | Positive windows with linked instructor speech | With explicit “correct the wrong action” speech |
|---|---:|---:|---:|---:|---:|---:|
| Train | 160 | 12,954 | 1,464 | 376 | 208 (55.3%) | 200 (53.2%) |
| Validation | 207 | 17,873 | 1,896 | 544 | 324 (59.6%) | 312 (57.4%) |

Only 4/540 verbally corrected validation actions and 1/335 training actions
lacked a resolvable correction-conversation link. The source archive used for
this audit contains train/validation annotations only. No official test labels
or media were read.

## Selected-checkpoint validation strata

All three arms always included the frozen task-text embedding; “audio-only” and
“video-only” mean the other sensor feature was masked. The selected checkpoints
are from the same already-observed validation set (audio step 320, video step
192, joint step 128). Metrics below are descriptive, post-selection validation
strata; there are no new model-selection decisions or confidence intervals.
Accuracy on positive-only strata is positive-class recall.

| Selected arm | All 1,896 labeled windows accuracy | Negative windows accuracy (n=1,352) | Positive windows with explicit correction speech (n=312): recall / NLL | Positive windows without explicit correction speech (n=232): recall / NLL |
|---|---:|---:|---:|---:|
| Audio-only | 0.7379 | 0.9105 | 0.4135 / 0.8305 | 0.1681 / 1.2402 |
| Video-only | 0.7131 | 0.9763 | 0.0769 / 1.1707 | 0.0345 / 1.2881 |
| Joint AV | 0.7305 | 0.8580 | 0.5545 / 0.8234 | 0.2241 / 1.3410 |

The negative-window majority baseline is 1,352/1,896 = 0.7131 accuracy; the
video-only selected checkpoint equals it. Audio-only and joint recall on
positive windows is much higher when explicit correction speech is in the
input. This is consistent with the readout using instructor speech as a
label-correlated cue. It does not prove the model decoded a particular
transcription, nor that video contributed useful evidence. On the positive
windows without explicit correction speech, all arms remain weak.

The earlier full-run report remains authoritative for aggregate selected
metrics, calibration and overfitting: HoloAssist joint AV did not beat
audio-only on aggregate validation, all three arms overfit substantially by
the fixed 2,048-step endpoint, and this task does not meet the 95–98% product
target. This stratification uses selected-checkpoint predictions from the
selection validation set and must not be described as independent or blind
quality evidence.

## Reproduction and disposition

~~~powershell
python -m scripts.audit_holoassist_window_label_alignment --annotations C:\CodexArtifacts\pretrained-reuse-r2\holoassist-source-screen-20261019-01\data-annotation-trainval-v1_1.json --selection C:\CodexArtifacts\pretrained-reuse-r2\holoassist-source-screen-20261019-01\video-pitchshifted-selection-160-207-seed17.json --event-dir C:\CodexArtifacts\pretrained-reuse-r2\holoassist-video-pilot-160-207-seed17-pitchshifted-attempt-01\event-corpus-v1 --window-dir C:\CodexArtifacts\pretrained-reuse-r2\holoassist-video-pilot-160-207-seed17-pitchshifted-attempt-01\temporal-windows-v1 --readout-root C:\CodexArtifacts\pretrained-reuse-r2\holoassist-video-pilot-160-207-seed17-pitchshifted-attempt-01\readout-attempt-01 --run-metadata C:\CodexArtifacts\pretrained-reuse-r2\holoassist-video-pilot-160-207-seed17-pitchshifted-attempt-01\readout-attempt-01\run-metadata.json --output C:\CodexArtifacts\pretrained-reuse-r2\holoassist-label-window-audit-05
~~~

The audit ran on CPU in about two seconds (Python 3.11.9); the existing RTX
3080 was occupied by an unrelated Orbitune process, so no HotUGPU reservation
or GPU operation was attempted. Existing HoloAssist files and model artifacts
remain read-only.

Do not retrain or re-evaluate the same selected validation set. A follow-up
must use a prospectively defined task that distinguishes visible action
mistakes from instructor feedback, with new session-disjoint train/validation
groups and a task-time cutoff set before corrective speech when autonomous
judgment is intended. Keep HoloAssist as a narrow task benchmark, not an
Omni quality claim.

