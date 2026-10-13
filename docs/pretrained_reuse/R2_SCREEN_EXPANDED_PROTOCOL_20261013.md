# Fixed expanded screen training

Compare original crop-only768->8 linear classifier with16 training apps rather
than8:1923 training regions rather than1017. Same723validation regions/4apps,
DINOv2 features/preprocessing, seed17, AdamW .001/.01, batch64,30epochs, earliest
minimum validation NLL includingepoch0. Separate selected/final checkpoints,
all predictions/history, class/app metrics, exact selected reload. No context
features, no new head capacity, no labels inferred for held regions.

This keeps epochs constant, not update count: expanded31 batches/epoch vs16,
930 vs480 optimizer updates. Train normalization and class mix also change.
Interpret as training coverage intervention, not isolated app-count causality.
Do not change the selector or try extra seeds if results fail. Existing files
remain immutable; output groundcua-expanded-classifier-01. Feature extraction
actually completed906x768 in19.79sec;194804224bytes peak GPU allocation, exact
saved reload, frozen22056576parameters. No training occurred during extraction.
