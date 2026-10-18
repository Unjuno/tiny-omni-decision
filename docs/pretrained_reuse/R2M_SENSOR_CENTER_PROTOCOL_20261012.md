# One-factor center readout diagnostic

Hypothesis: global pooling obscures center-sample activity at transitions.
Use identical temporal model parameters and initialization, replacing only mean
pooling with the feature at index T//2 (64 for128 samples). Two kernel5 layers
give a9sample receptive field there: less global context is an explicit tradeoff.
Same14/3 subjects, windows, center targets, seed17,12epochs, batch128, AdamW
0.001/0.01, earliest minimum validation NLL. No reweighting or data edits.
Compare with saved temporal-mean results; no repeat baseline training.
Separate training-only smoke and output directories, all prior artifacts retained.
Report full/mixed/unmixed and class-balanced metrics, not only transitions.
This is one further development comparison, not blind evidence. Stop this bounded
readout series after this experiment and return to missing R2-M native/fusion
coverage; no automatic architecture search or extra seeds.
