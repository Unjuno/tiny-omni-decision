# JAQKET text Decision source screen — 2026-11-24

## Disposition

**CANDIDATE, limited to the official developer-created Dev1/Dev2 question files.** JAQKET offers a Japanese variable-choice task closer to the product's supplied-option probability interface than the recent fixed-label Jamp NLI probe. It is text-only and knowledge-intensive; it does not qualify multimodal or broad world-understanding quality.

## Primary-source findings

The [official AI-O dataset page](https://sites.google.com/view/project-aio/dataset) describes JAQKET v1.0 as Japanese quiz questions with 20 candidate Wikipedia entity titles. It publishes 13,061 train rows, 995 Dev1 rows, and 997 Dev2 rows. The page says Dev1 and Dev2 were separated so that one could act as a pseudo-test during development. It also identifies data provenance by split: train questions are from abc/EQIDEN and restricted to non-commercial research redistribution; Dev1/Dev2 were commissioned from quiz authors and provided under CC BY-SA 4.0. The train split will not be downloaded or used.

The [official baseline repository](https://github.com/cl-tohoku/JAQKET_baseline) is Apache-2.0 code and supplies direct S3 URLs for `dev1_questions.json` and `dev2_questions.json`. The baseline also uses a 1.9 GB Wikipedia entity text file; this probe will not acquire that file. Inputs are the question and its supplied 20 entity-title choices only. This intentionally tests pretrained parametric knowledge and option scoring without retrieval.

## Proposed data protocol

- Use Dev1 only for development. Freeze a deterministic internal split by answer entity and normalized question, preserving every row and all 20 options. Audit question, target-entity, candidate-set, and normalized-text overlap before fitting.
- Compare the existing Ruri-v3-30m reference, the in-budget LINE Japanese DistilBERT first-four-layer candidate, and Tohoku BERT-base as a text-only teacher/reference. Keep identical option-pair formatting, frozen encoders, shared scoring-head shape, training schedule, and option-order controls.
- Select the model and checkpoint using Dev1 validation only. Save full per-option probabilities, NLL, Brier, ECE, accuracy, calibration, source-group uncertainty, option-order invariance, and option-only controls.
- Only after the selection rule and candidate are frozen, evaluate Dev2 once. Dev2 is public development data used as a pseudo-test, not a sealed audit; do not tune or repeat after opening it.
- Do not use the non-commercial train questions, hidden/private competition data, or Wikipedia article-body corpus.

Before training, verify the official Dev1 payload hash, schema, row count, target-in-options condition, duplicate questions, and split sufficiency. If Dev1 cannot support a source-disjoint internal validation or has target/candidate shortcuts, stop before fitting and report the exact issue. Do not substitute Dev2 into training after seeing it.

## Known limitations

- Only about 1,000 development questions are available for fitting and internal validation.
- Dev2 is not blind and shares a benchmark provenance; use it once only after freezing the candidate.
- Quiz/encyclopedic fact recall is one text task, not sensory grounding or multimodal Decision quality.
- The 20 candidate titles are available in the input; Wikipedia article content is not. This differs from the official retrieval-assisted baseline.
- A passing result on this task cannot establish the 95–98% four-modality target.

## References

- Official data and licensing: https://sites.google.com/view/project-aio/dataset
- Official baseline code: https://github.com/cl-tohoku/JAQKET_baseline
- Developer-provided Dev1: https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_01/dev1_questions.json
- Developer-provided Dev2: https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_01/dev2_questions.json
- Independent licensing summary distinguishing train from validation/test: https://huggingface.co/datasets/sbintuitions/JMTEB
