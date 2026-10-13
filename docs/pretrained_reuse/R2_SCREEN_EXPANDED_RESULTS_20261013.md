# Expanded screen training result

Fixed30epoch run completed, selectedepoch2 by minimum validation NLL. Same723
validation regions;1923training regions/16apps, frozen encoder,6152head parameters.

| Training | Accuracy | Balanced Accuracy | NLL | Brier | ECE |
|---|---|---|---|---|---|
| original8apps |.334716|.262416|2.102968|.808964|.094643|
| expanded16apps |.413555|.273142|1.936768|.754293|.055637|

All five descriptive metrics improve, but the original training-frequency prior's
Accuracy .486860 and NLL1.834577 remain better. Do not promote or claim95% screen
quality. No paired uncertainty yet; four observed validation apps are insufficient
for a broad conclusion. More training data, altered mix/normalization and greater
update count are coupled; no pure app-count causal claim.

Separate best/final weights, full history and all epoch predictions retained at
C:/CodexArtifacts/pretrained-reuse-r2/groundcua-expanded-classifier-01.
Selected reload exactly reproduces saved probabilities. Feature extraction pins
weights/source/file hashes in groundcua-train-expansion-features-01. Source commit
7443649; command `python -m scripts.train_screen_expanded`. CPU fit; no extra GPU
training or paid compute, audit data untouched. Ruff passed for both new runners.
This closes the fixed expansion comparison; broad screen/Omni quality remains unmet.
