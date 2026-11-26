# Matched wording-view learning results

Both 1,024-update runs completed with exact best reload and frozen-weight checks.
Parent order and fixed conditions match; step0/128/256 predictions are byte-identical.
Development validation only. No sealed audit, product promotion or general95% claim.

| Arm | Step | Selector | Original Accuracy | Curriculum Accuracy |
|---|---:|---:|---:|---:|
| control | 0 | 0.605480 | 0.635827 | 0.840820 |
| control | 128 | 0.617022 | 0.646457 | 0.854004 |
| control | 256 | 0.674499 | 0.631890 | 0.850586 |
| control | 384 | 0.660617 | 0.618898 | 0.860352 |
| control | 512 | 0.727891 | 0.616535 | 0.852051 |
| control | 640 | 0.783249 | 0.625591 | 0.852539 |
| control | 768 | 0.789487 | 0.623228 | 0.849609 |
| control | 896 | 0.791426 | 0.603937 | 0.850586 |
| control | 1024 | 0.805592 | 0.612205 | 0.831543 |

control: best step0; selected encounters0, unique parents0. Full run8192 encounters/5120 unique parents;3072 repeated parent encounters.
Timed 599.444s; peak allocated VRAM 3521245696bytes.

| Arm | Step | Selector | Original Accuracy | Curriculum Accuracy |
|---|---:|---:|---:|---:|
| wording | 0 | 0.605480 | 0.635827 | 0.840820 |
| wording | 128 | 0.617022 | 0.646457 | 0.854004 |
| wording | 256 | 0.674499 | 0.631890 | 0.850586 |
| wording | 384 | 0.657404 | 0.621260 | 0.856445 |
| wording | 512 | 0.726301 | 0.611024 | 0.851074 |
| wording | 640 | 0.785879 | 0.621654 | 0.847656 |
| wording | 768 | 0.794772 | 0.622835 | 0.844727 |
| wording | 896 | 0.801884 | 0.608661 | 0.836914 |
| wording | 1024 | 0.761508 | 0.616535 | 0.840820 |

wording: best step0; selected encounters0, unique parents0. Full run8192 encounters/5120 unique parents;3072 repeated parent encounters.
Timed 638.051s; peak allocated VRAM 3521245696bytes.

## Selected and endpoint metrics

| Arm | Checkpoint | Source | Accuracy | NLL | Brier | ECE |
|---|---|---|---:|---:|---:|---:|
| control | selected (0) | original | 0.635827 | 0.835591 | 0.470606 | 0.018915 |
| control | selected (0) | curriculum | 0.840820 | 0.375370 | 0.214324 | 0.016823 |
| control | endpoint (1024) | original | 0.612205 | 0.970316 | 0.523444 | 0.128927 |
| control | endpoint (1024) | curriculum | 0.831543 | 0.640868 | 0.268435 | 0.104263 |
| wording | selected (0) | original | 0.635827 | 0.835591 | 0.470606 | 0.018915 |
| wording | selected (0) | curriculum | 0.840820 | 0.375370 | 0.214324 | 0.016823 |
| wording | endpoint (1024) | original | 0.616535 | 0.925847 | 0.509677 | 0.090492 |
| wording | endpoint (1024) | curriculum | 0.840820 | 0.597168 | 0.251758 | 0.093254 |

Paired state bootstrap:10,000 draws, seed17, percentile95% intervals in JSON.
Selected checkpoint metrics reuse selection validation; they are not blind estimates.
Rolling training CE averages changing examples and is not a fixed train/validation gap.
Cloud cost0; local electricity unmeasured. Full source/config/artifact hashes in JSON.
No new image/audio/video measurement. Wording augmentation changes syntax and synonyms
jointly; no individual word-level causal claim. Existing artifacts remain preserved.

## Decision

Do not adopt either continuation. Both select the unchanged coverage1792
initialization (step0 here), with exact same best weight hash. Zero selected
encounters means zero ADDITIONAL updates in this experiment, not an untrained
model: inherited coverage1792 itself consumed14,336 encounters in its earlier
run, following older training.

At1024, wording-minus-control NLL improves by .04447 on original and .04370 on
curriculum; state-bootstrap95% intervals exclude zero. Brier also improves.
Accuracy deltas +.43pp / +.93pp have intervals containing zero. These endpoint
probability improvements do not beat the initial model. Both final models have
worse NLL/Brier/ECE than their shared start, despite early Accuracy increases.
This is evidence against adopting this repeated-small-state training schedule;
it is not a model-capacity ceiling or proof that paraphrases cannot help.

Total timed loops1237.495s (20.62minutes), including scheduled validation/reload,
excluding preparation/load/tokenization. Each arm:8,192 encounters,5,120 unique
parents; all are previously used training data. No new unique-data gain.

Next work should not extend this schedule blindly. Investigate genuinely new
compositional coverage or a better pretrained shared representation; preserve
the full multimodal objective and test a fixed intervention before scaling.
