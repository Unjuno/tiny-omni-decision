# R2 CLEVRER reasoning reconciliation and Storytime AV source screen

Date: 2026-10-23
Status: the previously approved CLEVRER reasoning run is complete; Storytime is **REVIEW / NOT ADMITTED**.

## Previously approved CLEVRER reasoning design

The requested fresh-scene CLEVRER explanatory, predictive, and counterfactual experiment had already been run and documented on branch `codex/r2-clevrer-fresh-reasoning-20261021`, commit `016face5ece2dd11835af641b1759104e1646a1a`. This reconciliation reads its committed status report and existing external receipts; it does not repeat training or evaluation. Full settings, split provenance, run hashes, failures, and metrics remain in the [original execution report](https://github.com/Unjuno/tiny-omni-decision/blob/codex/r2-clevrer-fresh-reasoning-20261021/docs/pretrained_reuse/CLEVRER_FRESH_REASONING_V1_STATUS_20261021.md).

The frozen train/development/holdout partitions contained 200 scenes each. Training used 2,632 binary decisions (200 scenes) for 24 epochs / 1,008 optimizer updates. The 295,169-parameter readout selected epoch 10 by minimum validation macro question-type NLL. The post-selection holdout was evaluated once: 2,843 decisions across 200 scenes. The split used CLEVRER official-train source records, VideoPrism-B frozen video features, and MiniLM frozen text features. The task was a candidate-statement binary decision; it was not audiovisual fusion.

| Holdout macro by question type | Legacy v0 | Candidate v1 | Candidate − legacy |
| --- | ---: | ---: | ---: |
| Accuracy | 49.33% | 49.66% | +0.33 pp |
| NLL | 0.6953 | 0.6934 | -0.0020 |
| Brier | 0.5022 | 0.5002 | -0.0020 |
| ECE-15 | 0.0313 | 0.0145 | -0.0167 |

Paired scene bootstrap (2,000 replicates, seed 17) gave the Accuracy difference a 95% interval of **-3.22 to +3.89 pp**. NLL, Brier, and ECE intervals also included zero. Per-type accuracy was mixed: explanatory 47.24% vs 49.68% legacy, predictive 50.42% vs 47.88%, and counterfactual 51.31% vs 50.41%. The validation-selected candidate was near binary chance and did not show a supported holdout gain. This holdout is spent; do not retune on it or present it as an ongoing blind audit. At the time of this run, the saved fetch report verified zero overlap with its 4,591-item historical media inventory and zero cross-partition media overlap.

The saved paired-prediction file was read-only rechecked: 2,843 rows, 2,843 unique IDs, 200 scene IDs, and finite normalized probability pairs for both models. Independent recomputation of Accuracy/NLL/Brier/ECE-15 by question type and macro-by-type matched the saved report within `1e-6`. Key artifact hashes:

| Artifact | SHA-256 |
| --- | --- |
| Development run report | `d8476c67626974e14ee35eb8bf65674e5a2a6e7e0b2e24848ac1c155c35eda4c` |
| Selected checkpoint | `e088af8c8effa2ff6a48c047c18b1610756cbe4e0644bbca6c6b36d91f7a05a6` |
| Effective config | `37147fca815f0c17f4433a80ef4f6db1ca49f53410fdad5300abc69605311693` |
| Holdout evaluation report | `f6cbca5f0798ba2aab2c0c43f90f3fbf70fc60cc86a884222dc96b0564027ae7` |
| Paired holdout predictions | `a3850e0dae3c0045a061a2efaa2af34bfe71c45cc50278cca64fbc7d5a524964` |
| Holdout rows | `de5a83d5b42f09678a68533089d8f22b98d1e53d8c57bfc53bf192a54891608e` |

## Storytime audiovisual-quality source screen

The official [QUALINET dataset page](https://qualinet.github.io/databases/audiovisual/the-storytime-dataset/) describes 160 simulated videotelephony clips: four German-language stories, each in ten consecutive parts, with four quality levels per part. It reports subjective scores, 2–11 ratings per clip, and CC BY 4.0. The authors' [publication page](https://voigt-antons.de/publication/2022-10-04-C67/) confirms the controlled videotelephony setting and the user study.

This could support a narrow same-content audiovisual-quality ranking task, but it does not establish a general semantic decision benchmark or useful multimodal complementarity. The 160 variants reduce to 40 underlying story-part groups. Keeping all four quality variants of a part together gives at most 40 content groups; holding out whole stories leaves only four top-level groups. That grouping limitation is an inference from the published structure and makes a strong source-independent estimate unlikely.

The dataset's precise rating schema, participant-level split, source-file provenance, and per-file hashes were not verified. The linked OSF record `https://osf.io/cyb8w/` returned HTTP 403 on 2026-10-23, so no archive contents or media were accessed. The rating fields and license coverage of the actual clips therefore remain unconfirmed in this screen. No data, media, model, or GPU resources were changed or used.

**Decision: REVIEW / NOT ADMITTED.** Do not use it for model fitting or quality claims until the OSF metadata/files are accessible, the rating schema and media rights are checked, and a grouped task protocol with unimodal controls is fixed. Any future result would be a narrow audiovisual-quality study, not evidence for general video reasoning or the 95–98% product target.
