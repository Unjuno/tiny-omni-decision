# Spoken-palette TRAIN-only modality interventions — 2026-11-22

## Status

This is a fixed-checkpoint mechanism diagnostic on the frozen Lombard GRID
TRAIN corpus. It tests whether the spoken-color plus screen pipeline reacts to
paired evidence when only one input is intervened on. It is **not** an
independent quality evaluation and does not pass the R2-M both-input-required
gate.

The frozen model was not trained, recalibrated, or selected in this work. No
validation, calibration, evaluation, or sealed-audit records were opened. Both
runs used CPU; `CUDA_VISIBLE_DEVICES` was empty. The Orbitune GPU process was
left running. Original corpus, checkpoint, and runtime assembly files were
read-only. All new outputs use separate C: attempt directories.

## Integrity fix and test

The first preflight found a grouping bug in the audit script: the official
source TRAIN layout contains 36 speaker groups, while the frozen task corpus
TRAIN split contains 24. The script had counted every official TRAIN speaker
instead of restricting identities to source IDs referenced by the frozen
corpus `train.jsonl`. It now filters by those corpus TRAIN IDs, then requires
exactly 24 groups and all four color labels in each. This avoids including
selection/calibration/evaluation speaker groups in the diagnostic. A regression
test covers that boundary and unknown source IDs.

The corrected preflight selected 96 unique audio/screen pairs from 24 training
speaker groups, exactly six examples in each color × target-position cell. It
matched these pinned inputs:

- Corpus manifest SHA-256:
  `b75a9e5ea629b8e1d8cb188ef4ee9c9a9b3e784b0f387ce9eed0a27072d72c7a`
- Frozen TRAIN JSONL SHA-256:
  `7c2988c5d688d89e605c8e9683cb5db1155faf127f6a2b3e61e1d72ad71aa170`
- Official source TRAIN layout SHA-256:
  `259f3ee36aa956618a89819338c2ecb67b194c82dbf636df70b551e00a5d78b0`
- Lombard audio archive SHA-256:
  `e36707602fdbeaa2e2284acb29bba883261d5e86857f694ac40cbb56cdcc5e10`
- Runtime assembly manifest SHA-256:
  `616b9b61e983d875a6a29e71d568f1ba0a6aeba6705da61ee66a88b4be43b3fd`

The failed preflight attempts stopped before inference and created no result
directory. The successful scripts fail closed on mismatched hashes and refuse
to overwrite an output path.

## Fixed-checkpoint counterfactuals

The first diagnostic ran 96 three-condition pairs through the unchanged
encoder-only Path A spoken-palette runtime. The screen geometry, audio, option
IDs, option order, and target coordinates stayed fixed within each pair.

| Condition | Support | Target-position accuracy | Mean maximum probability | Mean normalized entropy |
|---|---:|---:|---:|---:|
| Matched screen and spoken color | 96 | 96/96 (100%) | 0.99566 | 0.01385 |
| Target color removed from screen | 96 | not defined: no supplied option is valid | 0.73889 | 0.44232 |
| Unrelated distractor color removed | 96 | 96/96 (100%) | 0.99307 | 0.01952 |

Using `1 - maximum probability` to rank target-absent inputs against matched
and distractor-absent inputs gave pooled AUROC 0.98009; the macro speaker-group
AUROC was 0.97526. A 10,000-replicate, seed-17 paired bootstrap over 24 whole
speaker groups gave a pooled AUROC 95% percentile interval of `[0.96538,
0.99143]`. The paired mean maximum-probability change was −0.25678 for target
absence (95% interval `[−0.29590, −0.21858]`) and −0.00260 for distractor
absence (95% interval `[−0.00952, 0.00305]`).

The runtime still returns a four-option probability distribution when no
option is valid. This diagnostic does not add or validate abstention behavior.

## Fixed-screen audio rotation

A second intervention kept each of the same 96 screens and option lists fixed,
then rotated the audio to the next color from the **same speaker group**. The
correct option was derived from that color's position in the fixed screen. All
96 target positions changed; each audio source was used once and each spoken
color appeared 24 times.

The fixed checkpoint changed its selected option on 96/96 inputs and selected
the rotated audio color's screen position on 96/96. On these TRAIN examples,
Accuracy was 100%, NLL 0.00744, multiclass Brier 0.00101, equal-width ECE-15
0.00712, and mean confidence 0.99288. A 10,000-replicate, seed-17 bootstrap of
24 complete speaker groups gave 95% percentile intervals:

| Metric | Estimate | 95% interval |
|---|---:|---:|
| Accuracy | 1.00000 | `[1.00000, 1.00000]` |
| NLL | 0.00744 | `[0.00290, 0.01323]` |
| Multiclass Brier | 0.00101 | `[0.00018, 0.00218]` |
| Equal-width ECE-15 | 0.00712 | `[0.00283, 0.01256]` |

This demonstrates that the frozen runtime is sensitive to the audio condition
and the screen position in these paired TRAIN cases. Since the 24 speakers and
their task examples were used for fitting, the perfect scores and degenerate
Accuracy interval are in-sample results. They do not establish generalization,
natural AV quality, a reliable 95% product floor, or the R2-M gate.

## Runtime, size, and artifacts

Both inference runs used the same encoder-only modular assembly:

- 163,102,597 unique logical runtime parameters.
- 594,168,210 package payload bytes, excluding the package manifest. This
  remains above the provisional 200 MB package target.
- Whisper Tiny encoder revision
  `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`; extracted encoder state SHA-256
  `d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8`.
- The first 96-pair, three-condition run used 162.49 CPU seconds. The audio
  rotation used 76.38 CPU seconds. No cloud compute or GPU compute was used.
- Source commit for both inference artifacts:
  `22c42ff929e2248e20dcff1f6f933d0a1296e4d3`. The run-specific code hashes are
  embedded in each `result.json`; the worktree also contained the untracked
  audit scripts shown below.

Artifacts, outside Git:

- Counterfactual run:
  `C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-absent-target-audit-20261122-attempt-01\`
  - `result.json` SHA-256 `641126eea9c80824db7d111b7506f163ba82a6e2d23318e352ee36bddb1bcf0a`
  - `predictions.jsonl` SHA-256 `8bdc08e35e147acd182047d203e2740a27ba86479cc56585238c4f3e52040b6f`
  - `speaker-bootstrap.json` SHA-256 `a1d22d406eb6334cbfa04966525f04e3c6a3782da09ef352417d29bfe73597dc`
- Audio rotation:
  `C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-audio-swap-audit-20261122-attempt-01\`
  - `result.json` SHA-256 `744fa63ed0365fc63baf79a1508fadfb37a6b2d3299cfbd88c63bd416538b728`
  - `predictions.jsonl` SHA-256 `16f21fe03d770a0c8f547c816cbdfbc882f53ee19d1e9088fdd8fc9f33bb52fc`
  - `decision-metrics.json` SHA-256 `a1b80aba93e302e36668951cf551d45c3fbc6ef4a2782a244d7b50298e60a62a`

## Reproduction and verification

From the repository root, with the pinned Python 3.11 environment and the
read-only artifacts at the recorded paths:

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
python -m scripts.pretrained_reuse.audit_spoken_palette_absent_target `
  --output-dir "C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-absent-target-audit-20261122-attempt-01"
python -m scripts.pretrained_reuse.summarize_spoken_palette_absent_target `
  "C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-absent-target-audit-20261122-attempt-01\predictions.jsonl" `
  "C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-absent-target-audit-20261122-attempt-01\speaker-bootstrap.json"
python -m scripts.pretrained_reuse.audit_spoken_palette_audio_swap `
  --output-dir "C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-audio-swap-audit-20261122-attempt-01"
python -m scripts.pretrained_reuse.summarize_spoken_palette_audio_swap `
  "C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-audio-swap-audit-20261122-attempt-01\predictions.jsonl" `
  "C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-audio-swap-audit-20261122-attempt-01\decision-metrics.json"
```

Focused verification: `16 passed`; Ruff passed on the four audit/analyzer
scripts and four corresponding test files. The tests cover the source-ID
split boundary, deterministic paired selection, audio rotation, option-order
integrity, finite probability distributions, multiclass metrics and
group-bootstrap reproducibility. Full repository tests and GitHub Actions were
not run for this local diagnostic.

## Next gate

Keep this evidence as a mechanism check only. Before treating this as useful
joint Decision quality, evaluate the same frozen runtime on a predeclared
independent speaker/source group that was not used for fitting or checkpoint
selection, with matched, audio-swapped, screen-swapped, and unimodal controls.
Do not use the sealed audit for that selection. The current 163.10M / 594 MB
assembly, runtime parity issue, remaining modality/task gates, and product
quality targets remain open.
