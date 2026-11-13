# Frozen descriptive grounding: bounded result

## Execution and data

Source commit `71f3f02`; protocol `25772de`. Both fresh heads completed 24 epochs
with seed17 on 1,337 training questions / 256 scenes; full validation contains
2,200 questions / 200 scenes. The 1,475 question-text duplicates were removed
from training before execution, with zero validation deletions. This is a new
content-disjoint descriptive diagnostic, not directly comparable to prior binary
causal-task Accuracy. No sealed audit or model promotion.

Each head consumed 32,088 question encounters (1,337 unique, 30,751 repeats).
Selected video epoch6 consumed 8,022; selected no-video epoch2 consumed 2,674.
Best and final are separate; both best reloads reproduce logits exactly.
MiniLM and VideoPrism are frozen. A separate two-epoch smoke completed first.

## Selected validation quality

Equal weighting across five answer taxonomies; ECE uses 15 bins within taxonomy.

| Condition | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Video + question/options | 0.38137 | 1.21927 | 0.66254 | 0.04951 |
| Question/options, zero video | 0.35722 | 1.22337 | 0.66251 | 0.02566 |

| Taxonomy | Video Accuracy | No-video Accuracy |
|---|---:|---:|
| Count | 0.2650 | 0.2713 |
| Existence | 0.5200 | 0.5175 |
| Color | 0.1606 | 0.1549 |
| Material | 0.5578 | 0.5413 |
| Shape | 0.4035 | 0.3012 |

Scene-paired percentile bootstrap: seed17, 10,000 draws, equal taxonomy weighting.
Video-minus-no-video Accuracy = +2.414pp, 95% interval [-0.885,+5.729]pp;
NLL = -0.00409 [-0.01474,+0.00659]. These intervals include zero.

For the selected video head, matched-minus-cyclic-mismatched video Accuracy is
+2.316pp [+0.991,+3.628]pp; NLL -0.00954 [-0.01359,-0.00564], Brier -0.00533
[-0.00810,-0.00261]. This supports a **small scene-information contribution within
this diagnostic**, while failing to establish improvement over the independently
trained no-video model. Validation is observed and used for checkpoint selection;
intervals do not remove selection or repeated-experiment bias. No blinded claim.
No-video matched/shuffled predictions are exactly identical.

## Overfitting and decision

Video training NLL falls from 1.13580 at selected epoch6 to 0.77411 at epoch24,
while validation NLL rises from 1.21927 to 1.33995. Endpoint Accuracy rises to
0.41005 but ECE worsens to 0.17179; the predeclared NLL selector is retained.
No-video also overfits: selected train/validation NLL 1.18775/1.22337 versus
endpoint 0.87389/1.43192. Reject both as product candidates; do not extend epochs.

Unlike the causal-task head, this descriptive head shows a small matched-scene
signal. That is sufficient to justify a bounded question-conditioned token
readout comparison on these same inputs, not encoder unfreezing or a claim of
video reasoning. Pooled features and scene shuffling do not establish temporal
order understanding. Text/template transfer, small unique-scene count and
representation/readout capacity remain confounded.

## Resources and reproducibility

Frozen towers: VideoPrism 114,365,184 and MiniLM 17,389,824 parameters; head295,169.
A live video configuration would contain 132,050,177 parameters. This experiment
loads cached video features and MiniLM; it is not an assembled four-modality model.
No-video preserves head dimensions but video input weights have no influence.

Feature preparation3.70s; video loop24.32s; no-video loop22.16s, including repeated
metrics/reload/diagnostic work. Peak allocated CUDA memory125,369,344bytes excludes
live VideoPrism extraction. Local RTX3080 Laptop, no cloud cost; electricity not
measured. Concurrent CPU tests may influence these throughput observations.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/video-grounding-training-01`.
Command: `python -m scripts.train_video_grounding --output C:/CodexArtifacts/pretrained-reuse-r2/video-grounding-training-01`.
Existing output directories are rejected. Smoke used `--smoke` and a separate
`video-grounding-smoke-01` directory. Source inputs, exact revisions, hashes,
all curves, best/final hashes and per-example probabilities remain saved.
[Machine-readable report](R2_VIDEO_GROUNDING_RESULTS_20261012.json).

No old configuration, corpus, checkpoint or media was modified. This stage
remains far below the four-modality95% target; the objective is still active.
