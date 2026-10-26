# TinyCLIP Open Images fresh task-ID validation (2026-10-26)

## Scope and protocol

This is a task-specific frozen image-grounding diagnostic contributing to R2.
It is not the Path A integrated Decision candidate, a blind test, or a product
quality result. No training, prompt search, calibration, checkpoint selection,
or sealed-audit access occurred. The fixed prompt was `a photo containing
{label}.`; the model compared each image with four supplied labels.

The sample uses 220 newly drawn Open Images V6 validation image IDs from the
same 11 target classes used in the existing pilot. All 220 pilot validation
image IDs were excluded before sampling. Rights-page matching retained 174
image records; media preflight/download produced 168 usable images. Six
unavailable records were excluded without replacement. The 168 retained
images have zero ImageID and OriginalMD5 overlap with the page-matched training
set. This is a fresh task-ID subset of the same public validation split, not an
independent split; source-support aggregates had previously been inspected.

Public Flickr landing pages declared CC BY 2.0 with matching attribution for
the retained candidates. Individual ownership was not independently verified,
so this is a bounded page-rights screen rather than a legal ownership finding.
The sealed audit was not loaded.

## Result

Pinned TinyCLIP-ViT-39M-16-Text-19M-YFCC15M, revision
`07a4b0bc751cb64fecd2b661c048c1dd98d69444`, weights SHA-256
`112bdfb07ebc8994e9afba7a2a314e278493a675033595cf48e54cf8a17ae587`,
83,101,185 runtime parameters, declared MIT. CPU FP32; Python 3.11.9,
PyTorch 2.6.0+cu124, Transformers 5.6.2, four CPU threads.

| Input condition | n | Accuracy | NLL | Brier | ECE (15 bins) | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| Aligned image | 168 | 64.29% | 1.0385 | 0.5050 | 0.1365 | 0.7563 |
| Zeroed image | 168 | 43.45% | 1.1454 | 0.6613 | 0.1720 | 0.6051 |
| Deranged image | 168 | 33.93% | 1.7745 | 0.8907 | 0.2874 | 0.6198 |

Both image controls changed correctness on 79/168 rows relative to aligned
input. Aligned-vs-zeroed top-1 agreement was 38.10%; aligned-vs-deranged was
33.33%. This supports that predictions use image input on this task, while
64.29% accuracy and the calibration values do not meet the project's quality
goal. Per-target sample counts range from 11 to 19; the result is too small and
task-specific to generalize to arbitrary image decisions.

The earlier exposed pilot (same model and prompt, n=168) scored 64.88%
Accuracy, NLL 0.9902, Brier 0.4899, ECE 0.0942, and mean confidence 0.7409.
That comparison is descriptive only: the task-ID set is fresh, but it shares
the same public validation source and target classes.

## Reproduction artifacts

External result root:
`C:\CodexArtifacts\pretrained-reuse-r2\openimages-freshval-20261026-attempt-01\`.

| Artifact | SHA-256 |
|---|---|
| fresh candidate task manifest | `AD36BE1FCB65DF0BAF4D5433F5EB4CC6FC49316F133B015A5D8670956A234051` |
| retained validation media manifest | `AB531846397629D454Eebf5e36b1d69a5f5fc436a2d67861aac8e55ecb9e40b0` |
| frozen validation task manifest | `E426877F525AFE19EAF1CBCF16D36F60A63913F81992CB046600F97D1EF9F315` |
| result JSON | `1411AEEB9D92CC2C880C4759A7F7A803F163D1A682880DC9A67A55A396BCB148` |
| aligned predictions | `6DAF0A2C89F1A8ABDEF5DF9DFC254F7F49B0AE1117F759627DBF0704C2C88908` |
| control predictions | `B6D6C07BD290115F26764ECED15F39525AF099B1BD463BD26DAC7B95FEB0D73E` |

The model evaluation took 10.61 CPU seconds; controls took 20.87 seconds.
No GPU was used. At the subsequent status check, the RTX 3080 Laptop had
15,821/16,384 MiB allocated by a separate long-running process.

## Interpretation and next gate

This result adds a fresh-task-ID image reference and input-sensitivity control,
not a learned Decision readout. It does not satisfy the roadmap's common Path A
training/evaluation contract, four-modality task battery, independent-group
quality gate, or final package/runtime gate. Preserve it as a task-specific
reference. Further image model fitting should wait for the integrated Path A
training protocol to bind all required modalities and the chosen split policy.
