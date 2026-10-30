# Fixed prefix/vocabulary transfer after soft replay

Inference only, fixed final soft-replay checkpoint; all3072 previously observed
factorial questions preserved.128states x6questions x4conditions. Exact input
hash checked before evaluation. No prompt exploration, calibration or training.

| Body / prefix | Parent Accuracy | Soft replay Accuracy | Parent NLL | Soft replay NLL |
|---|---|---|---|---|
| familiar / absent |.9010|.9362|.2481|.1675|
| familiar / present |.5130|.8099|1.3596|.5242|
| domain / absent |.4310|.4297|1.1684|1.3347|
| domain / present |.3255|.2734|1.4635|1.6797|

Improved familiar-prefix robustness does not transfer to the domain wording.
The body intervention changes both vocabulary and syntax; do not attribute this
solely to word embeddings. Parent-to-soft changes include context training and
replay, so these results do not isolate the soft loss alone. Development inputs
were previously observed, not blind. No product promotion;95–98% remains unmet.
Stop extending the same context fit on the strength of in-template results.
Retain the parent and all negative results; future training needs a prospective
semantic-transfer intervention and an independent source/state contract.

Full Accuracy/NLL/Brier/ECE, checkpoint/input hashes, elapsed time and VRAM are
in the adjacent JSON. All recorded artifact hashes verified. Exact executed script
is preserved in the output directory; the repository copy only moves imports for
lint compliance after execution. No existing checkpoint/data was changed.
Output: C:/CodexArtifacts/pretrained-reuse-r2/soft-replay-prefix-factorial-01.
Command: `python -m scripts.evaluate_soft_replay_prefix`.
