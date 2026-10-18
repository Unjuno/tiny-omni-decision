# Comparison-vocabulary coverage control

The paired fixed-weight diagnosis isolated a large comparison-word effect.
Test a training-data intervention, not more steps on the unchanged recipe.
The existing general-quality and95% goals remain unmet; no smaller test replaces them.

Source is the frozen lexical-range train corpus. Preserve all8,192question IDs,
1,024states, values, labels, candidate order, row order and four sentence variants.
Change only the comparison operator in extrema/scalar-time questions: four pairs
minimum/maximum, smallest/largest, lowest/highest, least/greatest. Keep count and
conditional-selection questions unchanged. Seed151 assigns one pair per state
within each numeric-scale/sentence-variant stratum; retain all opposing questions.
Counts are1572,1548,1524,1500extrema/time questions respectively;2,048other questions
unchanged.4,572question strings change. No duplicated examples or new media.
Train SHA41b35f3c2b1161d21ebf123d8ba39b29aa17dc9dc82c5f03ff7f291b9d368c76.

These are known vocabulary weaknesses, so this is failure-guided development.
No previously evaluated state/record enters training. Do not describe future
tests on these now-trained words as held-out-vocabulary tests. Later generalization
requires independently fixed states/structures and honest seen-word accounting.
No sealed audit access, old probe rerun, favorable-word selection or redefinition
of existing scores.

Same parent encoder-coverage selected2048 initial weights
SHA4024788b1a9d17735612a98da6acb1a65af5d82a9d440b4fddb4ab62efb9d318;
do not initialize from later lexical/numeric runs. New AdamW, not exact resume.
Seed17,2,048updates,4original+4curriculum questions/update,LR2e-5constant,wd.01,
clip1,dropoutoff,all six existing encoder blocks/classifier trainable; embeddings
and pooler frozen. Total22,713,601;trainable10,647,169. No new parameters.
Same validation IDs/order, interval128, earliest min mean original/curriculum NLL,
including step0. No selector changes based on outcomes. Best and final separate,
frozen hashes and exact best reload mandatory. Fixed budget unless safety fails.

Separate2-update smoke and full run in new directories comparison-vocabulary-smoke-01
and conditional-repair-comparison-vocabulary-01. LocalRTX3080Laptop only, roughly
20minutes, no cloud. Existing artifacts/configs remain unchanged. Verify effective
config and sample order against lexical-range after launch. Record all curves and
calibration/retention regressions. No additional seed/budget authorized by this
protocol. This run alone cannot establish wording generalization or multimodal quality.
