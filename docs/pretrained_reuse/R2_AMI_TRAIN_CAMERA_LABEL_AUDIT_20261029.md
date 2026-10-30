# AMI train-only camera/label spot audit — 2026-10-29

## Scope and method

This is a read-only follow-up over the already acquired, train-only
ES2002a/ES2005a/ES2006a/IS1000a-d sample in
`C:\CodexArtifacts\pretrained-reuse-r2\ami-av-probe-20261021\train-av-sync-coarse-seed17-20261021-03`.
It uses the two existing target-camera contact sheets and the corresponding
train-only word-interval audit. No new media or annotations were fetched,
no validation/evaluation/sealed-audit material was opened, and no model or GPU
was used.

The visual review here is a qualitative screen by one reviewer. The target
camera and task labels are visible in the contact-sheet metadata, so this is
not a blind label adjudication, not an independent reviewer result, and not a
prevalence estimate. Its purpose is to record clear counterexamples to the
assumption that the designated camera always contains a usable view of the
labelled speaker.

## Integrity and speech-interval checks

- The preserved timing receipt contains 24 train clips from seven meetings,
  with 12 target-camera frame samples per clip (288 total). Requested-to-decoded
  PTS deltas have a 20 ms median and a 39.455 ms maximum.
- The paired train-only word-interval audit reports positive-duration target
  speech in all 24 clips, with mean target-speech coverage 1.0. Seven clips also
  contain at least some overlapping speech by another agent.
- Both examined counterexamples are direct joins on the frozen `sample_id`,
  not row-order guesses:

| Sample | Target camera | Clip | Annotated target speech | Visual screen |
| --- | --- | ---: | ---: | --- |
| `ES2002a-C-s17` | Closeup2 | 255.07–257.26 s | 2.19 / 2.19 s | No discernible target person/face in the sampled target-camera frames; a hand appears at the frame edge near the end. |
| `ES2006a-B-s17` | Closeup1 | 397.98–402.19 s | 4.21 / 4.21 s | Target-camera frames show the empty room; no target person is visible. |

These examples do not show that the transcript labels are wrong: the interval
audit supports that the labelled agent has speech in the crop. They do show
that the current camera stream can lack visible evidence for a labelled
speaker. Consequently, this sample cannot support an assumption that the
visual stream always contains the correct answer, and it does not establish
that audio and video are jointly necessary. Preserve these examples; do not
filter them based on this observation or use this screen as a model metric.

## Reproducibility receipts

- `result.json` SHA-256:
  `45fb9ea3ec815a991477cb39ca31dfe8bcd1ccac7bb1a42c9b183eb6a8615886`
- `train-sync-contact-sheet-01.png` SHA-256:
  `a7be89359024651b1e309bcb2ec98a09fa1697f2f99048c4a3181b5714122bd3`
- `train-sync-contact-sheet-02.png` SHA-256:
  `4c0149ad6ec37a786680ce218b5d57cb8fc9ada5e5f95ab7323a28145d79f646`
- `ami-target-speech-overlap-audit-20261022.json` SHA-256:
  `c0174176be975ca3b28fbb17225afdaa77249308b8a2ffb81f47f63a1496965e`

The existing timing record remains `NOT_ADJUDICATED`: mixed-headset RMS and
frame timestamp proximity do not prove speaker identity or lip-sync. The next
gate remains an independently reviewed, source-valid train/development task
with enough independent groups and paired unimodal/joint/perturbation controls;
this spot audit does not admit AMI for fitting.
