# Same-option quartet feasibility on train candidates

The [fixed exploratory procedure](R2_OPENIMAGES_BALANCED_QUARTET_FEASIBILITY_PROTOCOL_20261027.md)
enumerated all verified positive and three-negative combinations for the 257
already observed train candidates. It found 194 distinct four-label sets; 12
had at least one image for each of their four correct labels. Deterministic
greedy selection produced 15 disjoint quartets (60 unique images), leaving 197
candidate images unused by this diagnostic. No images were removed from the
original 257-row task or from any evaluation set.

Each quartet has the same ordered options on all four images and one target
per option. A predictor seeing only that ordered option list therefore gets
exactly one of four correct per quartet, regardless of its option preference.
This is a property of this constructed diagnostic, not evidence of model
quality or evidence that every other shortcut is eliminated.

The greedy selected target distribution is Door 10, Plant 8, Vehicle 2,
Clothing 6, Furniture 7, Building 14 and Window 13; Toy, Food, Sculpture and
Animal have zero. This small, uneven diagnostic cannot represent the full
11-class objective. The greedy method is deterministic but is not an optimal
matching algorithm; different methods could produce different coverage. These
counts were examined on train-candidate data, so this is exploratory and not a
blinded evaluation. No model was trained, no validation data was filtered,
and no source-rights or independent-evaluation gate was closed.

Private outputs are in
`C:/CodexArtifacts/pretrained-reuse-r2/openimages-balanced-quartets-20261027-attempt-01/`.
Raw SHA-256 of `summary.json`:
`586291d2eee86623e153f1c07c7f389f7d37367002554b78e662e7a646100765`;
`private-quartets.json`:
`12c9eca99a43c543c3dfcb4e2b565a7bdd7bf01731dc09d0314f44a2619ab901`.
The output pins the input, script and protocol hashes.
