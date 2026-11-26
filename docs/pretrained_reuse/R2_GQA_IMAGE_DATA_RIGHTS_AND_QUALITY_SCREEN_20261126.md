# GQA image-decision source and data-quality screen — 2026-11-26

## Disposition

**REVIEW_NOT_ADMITTED. No GQA question, answer, scene graph, or image payload
was downloaded or opened.** GQA is a promising image-plus-optional-criterion
Decision source because its official design exposes structured semantics and
question-type diagnostics. It is not yet eligible for model fitting: the
official pages inspected do not state a dataset-specific license, image rights
flow through COCO/Flickr sources, and cross-corpus image overlap with already
used Visual Genome data has not been measured.

## Pinned official-source evidence

- Official project: [GQA: Visual Reasoning in the Real World](https://cs.stanford.edu/people/dorarad/gqa/).
  The source is currently a web page rather than a pinned source-code release;
  no data revision was selected because no payload was acquired.
- The [official project overview](https://cs.stanford.edu/people/dorarad/gqa/index.html)
  describes 22M compositional questions, binary and open-answer forms,
  per-image scene graphs, structured question semantics, and balancing by
  question type to reduce language-prior shortcuts. It states that GQA images
  come from COCO and Flickr and that scene graphs are based on Visual Genome.
- The [official download page](https://cs.stanford.edu/people/dorarad/gqa/download.html)
  describes separate question, scene-graph, image, and feature resources. The
  page inspected contains no license statement. A third-party TensorFlow
  Datasets request lists CC BY 4.0, but that is not the publisher's grant and
  does not independently clear the underlying COCO/Flickr image rights.
- The Hugging Face [`lmms-lab-encoder/GQA` mirror](https://huggingface.co/datasets/lmms-lab-encoder/GQA)
  declares MIT and describes itself as a formatted GQA version for the
  `lmms-eval` evaluation pipeline. The page reports 24,206,801 rows and 30.1
  GB total. This is a downstream mirror, not the publisher's original
  dataset-specific grant; its MIT metadata cannot settle rights for the
  underlying COCO/Flickr images. The full mirror was not downloaded.
- The official [About page](https://cs.stanford.edu/people/dorarad/gqa/about.html)
  says questions were automatically generated from natural-language,
  crowdsourced scene graphs. This provides structured audit signals, but it
  does not mean each answer has independent human adjudication. Potential
  annotation errors and question-generation artifacts need measurement.
- The authors' [GQA paper](https://arxiv.org/abs/1902.09506) reports 89.3%
  human accuracy, 42.1% for a question-only blind LSTM, and 54.1% for a strong
  VQA model on its reported benchmark. Thus GQA can be a useful structured
  image-reasoning stress task, but not a standalone 95% acceptance benchmark
  for the product target. These published results are task-specific context,
  not this project's model measurements.

## Why the data may help, and what must be checked

GQA has stronger dataset diagnostics than a flat answer-only set: semantic
programs, image/question types, answer balance, and scene-graph references can
support checks for object, attribute, relation, and multi-step reasoning. Its
open-answer format is not automatically a valid multiple-choice Decision
benchmark. Any conversion must freeze plausible distractor rules, handle
synonyms and multiple valid answers, and preserve the original answer
semantics.

Before admission:

1. Obtain written publisher-level license evidence for annotations and
   clarify whether it covers derivative training use; audit COCO/Flickr
   rights per selected image rather than applying one blanket license.
2. Compare official image IDs and normalized questions against every existing
   Visual Genome/TallyQA image and corpus manifest in this repository. Exclude
   previously observed assets and related questions before defining new
   train/validation groups.
3. Partition by image identity, with semantic question/equivalence groups
   held together where the official metadata exposes them.
4. Measure question-only, answer-prior, option-position, and scene-graph-only
   baselines before fitting an image encoder readout.
5. Audit generated-question and scene-graph label correctness on a bounded,
   predeclared train-only sample; keep any fresh validation untouched until the
   protocol and candidates are frozen.
6. Use per-question-type Accuracy/NLL/Brier/ECE and grouped intervals. Do not
   let aggregate performance hide weak spatial/relation or compositional
   cells.

## Sources

- [Official GQA overview](https://cs.stanford.edu/people/dorarad/gqa/index.html)
- [Official GQA About page](https://cs.stanford.edu/people/dorarad/gqa/about.html)
- [Official GQA download page](https://cs.stanford.edu/people/dorarad/gqa/download.html)
- [Secondary TensorFlow Datasets license reference](https://github.com/tensorflow/datasets/issues/300)
- [Existing project TallyQA/Visual Genome source screen](R2_TALLYQA_VG_AMT_SOURCE_SCREEN_20261110.md)

This is a metadata-only screen. It did not access publisher validation/test
labels, model weights, external media, or sealed-audit data. No new training
or evaluation is authorized by this screen.
