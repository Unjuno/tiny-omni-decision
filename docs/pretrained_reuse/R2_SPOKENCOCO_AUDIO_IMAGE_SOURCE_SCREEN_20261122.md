# SpokenCOCO audio-image source screen — 2026-11-22

## Decision

**Do not admit SpokenCOCO for product-directed Path A fitting or evaluation.** No data or media was downloaded. This is a source screen only.

The source is structurally relevant: MIT SLS describes roughly 600,000 human-recorded English readings of MS-COCO captions, one reading per caption. A prospective supplied-option image retrieval task could pair a spoken caption with candidate COCO images. This would be natural audio plus image, but not a video task.

The rights gate fails under the repository's current source policy. MIT SLS publishes the spoken-caption corpora under CC BY-SA, while the repository's explicit permissive set does not include CC BY-SA. Separately, the COCO Terms state that the consortium does not own image copyrights and that image use must follow Flickr terms. The screen did not establish per-image permission for the assets that would be selected. Therefore the audio and image components do not jointly pass the existing fail-closed policy. Do not broaden that policy or infer an image license from the COCO annotation license.

The official landing pages also do not establish a verified group-disjoint task split for this experiment. Before reconsideration, the audio-to-image rows would need a pinned metadata inventory that proves audio speaker/source groups, image IDs, annotation/content overlap, and train/selection/calibration/evaluation separation. No such data inventory was built.

## Pinned page evidence

- [MIT SLS Spoken Audio Captioning Corpora](https://sls.csail.mit.edu/downloads/placesaudio/) states the SpokenCOCO scale/content and CC BY-SA license for the corpora.
- [COCO Terms of Use](https://cocodataset.org/dataset/termsofuse.htm) distinguishes CC BY 4.0 annotations from images whose copyright remains with their original owners and requires Flickr terms to be followed.
- Retrieval date: 2026-11-22. SHA-256 values for the fetched official HTML are in the machine-readable receipt.
- No audio archive, COCO image, annotation archive, model, evaluation split, or sealed audit was accessed.

## Reconsideration gate

Reconsider only if (1) the audio component is covered by a license admitted by the current policy or a separately reviewed policy decision, (2) every selected image has verified use rights from its original source, and (3) immutable source-group metadata supports a clean asset- and speaker-disjoint decision task. The current result supports none of those gates. See [`spokencoco-audio-image-source-screen-20261122.json`](../../manifests/pretrained_reuse/spokencoco-audio-image-source-screen-20261122.json).
