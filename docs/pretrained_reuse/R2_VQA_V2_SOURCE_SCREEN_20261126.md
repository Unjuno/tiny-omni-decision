# VQA v2 source screen — 2026-11-26

## Disposition

**`REVIEW_NOT_ADMITTED` for a product-directed Path A fit.** VQA v2 has a potentially useful natural image-plus-question task and large publisher train/validation splits, but the image rights are not covered by the VQA annotation license, per-image rights were not audited, and the v2 release does not provide a frozen four-option contract with a unique correct answer for every question. Do not download the corpus or images, fit a Decision readout, or claim an R2 pass from this screen.

## Official source facts checked

- The [official VQA dataset page](https://visualqa.org/) describes image-question answering and reports 204,721 COCO images in VQA v2, 1,105,904 questions, and 11,059,040 ground-truth answers (10 human responses per question). The [official download page](https://visualqa.org/download.html) lists 443,757 train, 214,354 validation, and 447,793 test questions and says downloading the dataset constitutes agreement to the [VQA Terms of Use](https://visualqa.org/terms.html).
- VQA v2 is documented as an **open-ended** task. Its annotation schema includes ten free-form answers and a `multiple_choice_answer` field equal to the most frequent ground-truth answer; this is not itself a per-example option set or proof that the answer is unambiguous. Using all human answers as soft targets or constructing distractors would require a predeclared Decision contract and a label-ambiguity analysis.
- VQA Consortium annotations are CC BY 4.0. The [official COCO terms](https://cocodataset.org/dataset/termsofuse.htm) state that COCO does not own image copyrights and that image use must follow Flickr Terms of Use; users accept responsibility for use of the images. The VQA annotation license therefore does not establish product-use rights for the linked images. A per-image rights filter might be investigated separately, but was not established here.
- Official VQA split counts were read from the public website only. No train/validation image-ID disjointness, repeated-question overlap, image-license distribution, or source-image rights were verified. Do not infer split integrity from question counts.

## R2 fit and next gate

VQA v2 could test a narrow image-plus-text question task and could help compare a compact text encoder with the existing text tower when paired with the same visual features. It does not test text-only, image-only, video, audio, sensor, or naturally synchronized multi-input necessity. The task would need a fixed candidate-construction rule, human-answer-aware targets, image-group splits, per-image rights evidence, and input ablations before it could be admitted.

For now, keep VQA v2 outside the experiment corpus. Reconsider only if the repository's license gate can admit the exact image subset and a source-grouped option-probability protocol can be frozen without using the official test. This screen accessed only the official public pages; no dataset payload, image, annotation row, model, or sealed audit was accessed. No GPU or cloud compute was used.