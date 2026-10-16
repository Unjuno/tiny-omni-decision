# Input replacement inventory and real-feature mechanics

**Inventory and mechanics pass; selected-model sensitivity remains unmeasured.**
The [prospective protocol](R2_GREATEST_HITS_INPUT_REPLACEMENT_PROTOCOL_20261017.md)
was fixed before inspecting matched-run validation outputs. Source commit:
`7c33713d54b70961bbb02e4b0f5ea7abe128fe3d`, `codex/r2-av-transport`.
No training setting, corpus, checkpoint or selector was changed.

## Executed metadata work

Generated one label-blind donor assignment for every one of the existing 8,875
development-validation questions. Donors have the same task, question and complete
options but come from a different recording date, parent and asset. No question
was removed. Reversing input order and changing every target label leave the
assignment unchanged. An independent implementation recomputed every date/donor
choice and its hash; all 8,875 agree.

| Task | Questions |
|---|---:|
| Action | 2,987 |
| Material | 2,978 |
| Reaction | 2,910 |

There are 3,693 unique donor questions; maximum reuse is 14. Both original and donor
sets cover all 101 validation recordings and all five dates. Donor reuse and
questions sharing recordings make these dependent observations, not 8,875
independent trials. Date distributions are intentionally recorded:

| Date | Original questions | Donor assignments |
|---|---:|---:|
| 2015-02-16 | 852 | 1,977 |
| 2015-03-13 | 508 | 2,113 |
| 2015-03-25 | 2,258 | 1,689 |
| 2015-09-28 | 312 | 2,094 |
| 2015-10-06 | 4,945 | 1,002 |

Each original excludes its own date before donor selection. The changing date mix
is part of this registered sensitivity intervention, not a matched-distribution
estimate of real corruption robustness. Source-label agreement counts are saved
separately; they do not establish correct labels for mixed recordings. Unchanged
predictions can reflect redundant donor information as well as an ignored input.

Planned views are natural, self-audio, self-video, donor-audio and donor-video:
44,375 forwards per selected arm, not yet executed. Existing natural validation
quality remains the original Accuracy/NLL/Brier/ECE evaluation. Donor views use
prediction flips, probability total variation, Jensen-Shannon divergence and
confidence change without assigning counterfactual correctness.

## Executed real-feature mechanics

Separately selected three fixed TRAIN pairs, one per native task, from the already
verified 21-event smoke features. This test does not use the validation inventory
for model inference. Loaded the five exact fitted-baseline initial states from
the preceding preflight and ran 75 CPU forwards through the new replacement
transport. It retains the original query and selected blocks' values, positions
and masks; speech and event-audio move together.

All 30 self-replacement checks reproduce natural logits exactly. The five initial
arms agree on all natural and replaced logits, as expected from their shared
baseline and zero final residual. This proves neither trained gate utility nor
counterfactual quality. No optimizer or encoder update, no validation forward,
no test/calibration/sealed payload and no cloud use occurred.

Inventory generation took 5.2421 seconds. Independent inventory audit plus actual
CPU mechanics took 51.8072 seconds. Both processes exited 0. The original full
video producer continued without intervention; at publication preparation it
reported 7,484 of 11,151 events across 288 of 409 parents.

## Evidence and reproduction

Run from the implementation commit with the existing Python environment and CUDA
hidden:

```powershell
python -m scripts.freeze_greatest_hits_replacements --output <new-output-directory>
```

The original output already exists and must not be reused. Exact interpreter,
command, source, input hashes and timing are in the receipt. Paths:

- `C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-replacement-inventory-01/`
- `C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-replacement-mechanics-01/`
- `C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-replacement-verification-01/`

| Evidence | SHA-256 |
|---|---|
| Inventory receipt | `f12690080072e7cb0c60e0528394367114afc296f8c9cbe3d281425937687c62` |
| Canonical paired assignment | `4960fb4829c9f87df832ae2044b06ef96cca62d65ce4ff56f62374372eac58c4` |
| Independent audit/mechanics receipt | `aa6df265a36b25a11f1845aa2736bb3dc7f502a13afc9cb7c92114f6b3691624` |
| Independent audit script | `6c371b09b44f692836766bdf9fdca6f6fcf7acd3b58471088498af46c71e0777` |

Twelve new tests cover deterministic/label-blind matching, complete scope,
no-donor rejection, exact block/query preservation and target-free sensitivity.
Eight initially failed for missing implementation; two additional regression
failures exposed empty/3-D score acceptance and were fixed. Final suite:
2,388 passed, 1 skipped (68.01 seconds); Ruff and 25 manifest commands pass.
Logs preserve observed failures. Exact pushed-commit CI is recorded in `ci.json`.

Remaining work: complete full video admission and measured GPU cost, run the five
fixed long curves, then score the frozen replacement inventory on the selected
checkpoints. Product complementarity, robust calibration and the 95–98% task
objective remain unproven. No model is promoted by this mechanics result.
