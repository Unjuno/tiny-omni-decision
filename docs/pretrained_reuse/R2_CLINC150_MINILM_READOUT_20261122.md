# R2 CLINC150 Frozen MiniLM Readout (2026-11-22)

## Scope

This is a second, multi-domain text decision probe for the pretrained-reuse
work. It tests whether compact frozen MiniLM encoders plus a small linear
head can classify fixed intent labels, including an out-of-scope class. It is
not a multimodal result, an arbitrary-option JSON benchmark, or evidence that
the 95% target generalizes beyond this dataset.

## Data and split

The source is `clinc/clinc_oos`, config `plus`, pinned at
`155b9c710419136e17307b80d0a13e68cd46b4ec`; the upstream repository is pinned
at `828f8093932c8fe6ca7936c3d2e52903b1c523de`. The source declares CC BY 3.0.
Only publisher train and validation files were fetched. The official test
file was not downloaded or opened.

The combined 18,350 source development rows contained 18,347 exact normalized
text groups. Three groups crossed the publisher's train/validation boundary;
two of those groups have conflicting labels. No rows were removed. A
deterministic seed-17, label-tuple-stratified 80/20 split assigned every
NFKC/casefold/whitespace-normalized exact-content group wholly to one side:

| Split | Rows | Unique content groups |
| --- | ---: | ---: |
| Train | 14,680 | 14,677 |
| Validation | 3,670 | 3,670 |

Cross-split normalized-content overlap is zero. The split manifest SHA-256 is
`229a890f1921ebd5bd50de31ef704baef733d6b4c44d6ee6396ad82fc4078c6d`.

## Fixed experiment

Both candidates used seed 17, CPU FP32, the same train/validation rows, mean
and standard-deviation fitted on train features only, a 151-class linear
readout (58,135 trainable parameters), AdamW (LR 0.01, weight decay 0.01),
batch size 256, up to 60 epochs, patience 6, min delta 1e-5, and minimum
validation NLL checkpoint selection. No calibration was applied. The
encoders remained frozen. Each selected epoch was 6; 12 epochs were run.

| Frozen encoder | Encoder parameters | Accuracy | Macro F1 | NLL | Brier | ECE-15 | CPU elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM-L6 | 22,713,216 | 95.72% | 95.97% | 0.17973 | 0.06457 | 0.01192 | 34.10 s |
| MiniLM-L12 | 33,360,000 | 95.72% | 95.98% | 0.17039 | 0.06211 | 0.01050 | 61.43 s |

Both scored 24/24 on the OOS validation examples; the small OOS subset is not
enough to establish robust open-set rejection. L12 ties L6 on accuracy and is
slightly better on the other listed point metrics, at 10.65M more frozen
parameters and about 1.8x the measured CPU time. These are one-seed,
checkpoint-selected development-validation results, not test estimates.

## Artifacts and limits

The full predictions, per-class metrics, training histories, selected heads,
and run metadata are held outside Git under
`C:\CodexArtifacts\pretrained-reuse-r2-local\clinc150-minilm-{l6,l12}-frozen-readout-v1-attempt-01`.
Source and grouped split files are also outside Git. The model and data hashes,
revisions, and safety flags are recorded in
`manifests/pretrained_reuse/clinc150-minilm-readout-20261122.json`.

The benchmark is single-turn English intent classification. It does not
demonstrate temporal reasoning, sensor/video/audio fusion, or general decision
quality. CLINC does not expose reliable person/source identity groups, so
speaker-level independence could not be checked. The shared development
validation was used for checkpoint selection and this L6/L12 comparison; no
claim of independent confirmation is made.
