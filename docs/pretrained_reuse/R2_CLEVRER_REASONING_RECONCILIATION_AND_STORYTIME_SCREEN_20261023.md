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

At the time of the initial screen, the OSF web page returned HTTP 403, and the rating schema and file inventory were still unverified. A read-only public OSF API follow-up later on 2026-10-23 resolved those metadata questions without fetching any video. The node is `cyb8w`, last modified 2024-02-28, and its license relationship explicitly returns **CC-By Attribution 4.0 International**. The complete metadata-only file inventory, including OSF-reported SHA-256 values, file IDs, and byte sizes, is preserved in [the OSF metadata manifest](STORYTIME_OSF_METADATA_MANIFEST_20261023.json). It lists 160 MP4 files (1,155,739,709 bytes total; 160 distinct source SHA-256 values), two short PDFs, and one score CSV.

The 20,258-byte `Selected Clip Scores.csv` was fetched into memory from its public download endpoint and not saved as a dataset artifact. Its actual SHA-256, `03b56852f9f223b742d6e30dfe23dca0e93732d54416e1110849396dc4775c0a`, matches OSF's file metadata. It has 160 rows and columns for the story, part, four-level quality condition, objective media-quality indicators, and aggregated subjective overall/video/audio scores. Each row records only 2–11 ratings; the CSV contains no participant-level ratings or participant-level split. Every one of 40 `story × part` groups has exactly four quality variants.

These data confirm a narrow, potentially useful audiovisual quality-prediction task. They do not resolve whether the public license covers every depicted/speaking person's privacy and publicity interests or whether the available aggregates support a defensible independent evaluation. For generalization, splitting by story-part gives 40 groups but can leave the same four stories and speakers on both sides; holding out whole stories leaves only four top-level groups. That grouping limitation is an inference from the published design and inventory. The split and privacy/consent limits make the source **REVIEW / NOT ADMITTED** for model fitting or product-quality claims at this time. No MP4 or PDF was downloaded or opened; no model or GPU resource was used. The score CSV was the only payload fetched and was inspected in memory.

Reproducibility note: the metadata manifest is a snapshot of OSF's public API, not a local media-hash verification. It includes the source-reported hashes so a future separately authorized acquisition can verify each file without trusting filenames. The score CSV's content hash was independently checked against the API metadata.

**Decision: REVIEW / NOT ADMITTED.** The source-level CC BY 4.0 declaration, score schema, and file inventory are now verified. Before any model fitting, resolve participant/privacy scope and define a grouped protocol with unimodal controls that can support an honest uncertainty estimate. Any future result would be a narrow audiovisual-quality study, not evidence for general video reasoning or the 95–98% product target.
