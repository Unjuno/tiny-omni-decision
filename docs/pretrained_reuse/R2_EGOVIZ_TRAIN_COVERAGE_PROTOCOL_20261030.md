# EgoViz training-session coverage comparison — protocol

The matched 24-session baseline overfit before 5,632 updates. Test whether
more distinct sessions help at the **same optimizer update budget**, before
adding more capacity. The development set has been observed repeatedly; this
is exploratory, and no sealed media or prior pilot development examples may
enter training.

The pinned public index and prior plan identify 80 reserved groups: all 16
prior pilot groups and the 24 train, 20 development and 20 sealed groups from
the previous plan. Select 24 previously unused groups outside all 80. The
metadata selection is deterministic: seed 20261030, two sessions from each
available category in rare-category order, then fill by least represented
category; choose the smallest eligible clip and use a seeded SHA tie-break.
Clips must be at least 15 seconds and at most 1 GB. Eleven categories have
eligible groups; `cooking_food` has no eligible group under these rules and
is not fabricated. The 24 files total 7,299,677,559 bytes in the index. Index
SHA-256 is `c83126637cc24a7ac2b3600724b50cf9c53966b9b098acc57d5534e14ade1074`;
the new train-only plan SHA-256 is
`18958dcece44ae8a387895cbb5b64ffeb218f4988a8d59401522d797f8703b3`.

Acquire one file at a time with the existing source size/ETag checks, compute
SHA-256 and MCAP CRC, construct timestamp-only four-choice windows, and extract
frozen VideoPrism and MOMENT features on CPU using the existing pinned model
revisions. Validate each package before releasing only this run's verified
temporary raw file. Keep a minimum 3,000,000,000-byte free disk reserve beyond
the next payload, source receipts, corpus and feature hashes. The source index
pins metadata; source media hashes are first recorded during acquisition, so
this is not a claim that each media SHA was known in advance. Preserve all old
corpora, raw data and checkpoints. Do not download or open sealed media.

Add the new train windows to the original 332 training windows, without
changing their feature values, labels or session grouping. The original 307
development examples, IDs, order and preprocessing stay fixed. Validate
cross-split group, chunk, sample and raw-media hash separation, feature hashes,
and every candidate/target correspondence before fitting. The candidate is the
original uniform-channel 81,920-parameter readout from fresh seed 17 weights;
use AdamW LR 0.003, weight decay 0.01, batch size 32 and four CPU threads.

Train exactly 5,632 updates, with no performance-based early stopping. Use
seeded shuffled full passes through the expanded training examples and consume
the final partial pass only as needed. The new sample order necessarily differs
from the smaller-corpus baseline. Record actual full-pass and partial-pass
exposures, per-session counts, unique example IDs and the order hash. Evaluate
at update 1 and every 88 updates, matching the 8-epoch evaluation cadence of
the original baseline (11 updates per original epoch). Preserve separately
the best development-NLL checkpoint (minimum delta 1e-6) and fixed updates
352, 704, 1408, 2816, 4224 and 5632; save per-example validation probabilities.
Stop for nonfinite loss/gradients or integrity/resource failure, but do not
shorten the budget for ordinary metric variation.

Compare the expanded run to the existing fixed-budget baseline on the same
307 development IDs at matching update counts, with paired 20-session bootstrap
(2,000 draws, seed 17) for Accuracy, NLL and Brier. Report best checkpoints
separately; selected-checkpoint intervals do not correct for selection. Report
actual sample counts, modality/category mix, loss curves, end/best checkpoints,
local time, memory observations and source hashes. Reload saved checkpoints
in a fresh process. This cannot establish blind performance, participant-level
independence, a full four-modality product, or grokking impossibility. Promote
nothing and use no paid compute.
