# Unmatched cross-source interface controls

Repeat the [previous real-media interface smoke](R2_GRID_CROSS_SOURCE_INTERFACE_PROTOCOL_20261028.md)
with the identical first Lombard training audio, first four GRID smoke videos,
frozen weights, preprocessing and calibration. Prespecify these controls:

1. Reorder the four video states together with their option IDs; the JSON
   probability mapping must exactly equal the original.
2. Remove audio, then remove all video; each fixed missing-evidence policy must
   return four probabilities of 0.25.
3. Swap the first two video states while keeping their option IDs fixed; record
   the absolute probability change without requiring it to increase or decrease.

Check content hashes, finite output, unchanged source files and CPU-only
execution. This is a deliberately unmatched interface diagnostic, not a
labelled task, quality evaluation, model selection or proof of semantic fusion.
No GRID audio, validation, sealed audit or new training is used.
