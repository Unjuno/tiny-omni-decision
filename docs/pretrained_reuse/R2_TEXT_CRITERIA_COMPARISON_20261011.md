# Binary criteria preservation comparison

Source commit 5ef346a; same pinned MiniLM and frozen text splits, seed17,
8 epochs, question_pair contexts, 512-token option limit, batches64,
AdamW .001/weight-decay .0001 and validation-NLL checkpoint selection.
The only configured difference is --preserve-noul-criteria. The local runner
uses fresh output directories text-criteria-control-v1 and
text-criteria-preserved-v1 under C:/CodexArtifacts/pretrained-reuse-r2/.
Original normalizer, baseline artifacts and corpus files remain unchanged.

Noul definitions had been dropped for 9,205 train and 987 validation decisions.
The option now attaches their definitions to true/false candidate text and
remaps the target consistently. Nested definitions retain their canonical JSON.
One validation candidate needs315 tokens, within the model's512-position limit.
Overlength candidates fail before feature extraction in criteria-preserving mode.
The control uses the same512 limit to avoid confounding that change with criteria.

| Selected epoch3 | Accuracy | NLL | Brier | ECE15 |
| --- | ---: | ---: | ---: | ---: |
| Control | .48819 | 1.02788 | .59373 | .02917 |
| Criteria preserved | .50906 | 1.01490 | .58347 | .03778 |

Accuracy, NLL and Brier improve modestly; ECE worsens. The input repair does
not establish sufficient Decision quality or semantic generalization. No
further seed search, release promotion or sealed evaluation followed. This is
single-seed development evidence with validation checkpoint selection.

Both runs completed on the local RTX3080 Laptop. Histories, pinned hashes,
feature/training times and peak allocated VRAM are in the adjacent full JSON.
Ruff passed; CPU suite314 passed,1 skipped. Weights and full feature caches
are local only; no cloud compute was used.
