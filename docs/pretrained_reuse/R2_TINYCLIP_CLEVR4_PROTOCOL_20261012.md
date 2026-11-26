# Fixed TinyCLIP CLEVR-4 grounding diagnostic

Evaluate pinned TinyCLIP83,101,185parameters without training/calibration on all
512 existing development validation images, four taxonomies, ten options each.
Do not use this previously observed set as a blind audit or tune prompts on it.
No GPU execution while the lexical-range training job is active.

Use the exact upstream labels/options, current pinned CLIP processor, FP32 and
native saved logit scale. No prompt ensemble or temperature search. Fixed text:

- color: `An image of <option> objects.`
- texture: `An image of objects with a <option> texture.`
- count: `An image containing <option> objects.`
- shape: `An image of <option>-shaped objects.`

Compare correct image, image features rotated by one position in saved ID order,
and uniform probabilities. Rotation is a bijection with no same-image assignment;
same-class assignments may occur and are retained. Freeze all512labels and media
hashes before inference. Hash-check each image; preserve all questions/options.
Record per-taxonomy Accuracy/NLL/Brier/ECE, macro metrics, all probabilities,
ordered IDs and assigned image IDs, load/count/time/VRAM. No option-count reduction.

Labels SHA3c213f13ef2214baf1130cbe49c1d981b0125bbff3813e4ac5b4e341e443d4a1.
Media manifest SHA77faf433844be6b350cd0205fd3834cd8b028a6491ddd9eae0f6e9b7610cd38a.
Weight SHA112bdfb07ebc8994e9afba7a2a314e278493a675033595cf48e54cf8a17ae587.

This measures image–text grounding over a synthetic taxonomy. The option
vocabulary identifies the task; arbitrary question comprehension is not established.
Comparing against the trained VideoPrism readout is a reference only because data
exposure, preprocessing, objectives and architecture differ. Image encoders also
do not establish temporal video reasoning. No final audit, product promotion or
multimodal95% claim. New output directory only; preserve original models/media.
