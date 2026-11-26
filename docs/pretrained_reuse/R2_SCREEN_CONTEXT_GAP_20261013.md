# Screen context fixed-training gap and independent reload

A separate CPU process verified all saved input and arm-artifact hashes, strictly
loaded both best and final heads, and exactly reproduced each saved validation
probability array. Fixed full training-set metrics were measured after loading;
these are not rolling minibatch metrics. No training or model selection occurred.

| Arm/checkpoint | Epoch | Train Accuracy | Validation Accuracy | Train NLL | Validation NLL |
|---|---|---|---|---|---|
| crop/selected |1|.6146|.3237|1.2702|2.0836|
| crop/final |30|.9282|.3444|.3133|2.6245|
| context/selected |0|.1426|.1300|2.2150|2.3122|
| context/final |30|.9892|.2296|.0919|6.3775|

Both heads fit training examples while validation proper scoring worsens. The
context arm has particularly severe overfitting. This rules out failure to fit
these training examples as the main observed symptom. It does not distinguish
app/style shortcut learning, sample size, semantic feature mismatch or label noise.
Do not increase epochs/capacity on this same pilot, or silently switch selection
to final Accuracy. Preserve the NLL-selected checkpoint, including contextepoch0.

The next admissible screen experiment must address representative app diversity
and semantic feature suitability with a new fixed task/data contract. The current
oracle-region classification is a prerequisite diagnostic, not action selection.
Global Omni quality is still unmet; no model promotion/compression is justified.

Reproduce: `python -m scripts.replay_screen_context_gap`. Adjacent JSON includes
all class metrics, hashes and independent-process parity for each checkpoint.
Ruff passed. Existing weights, normalization and all predictions remain unchanged.
