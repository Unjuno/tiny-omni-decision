# Paired wording controlled learning protocol

Frozen before model runs. Two seed-17 runs: control and wording, each 1,024
updates, eight questions/update (four original and four curriculum), AdamW
2e-5, weight decay .01, clip1, dropout off, deterministic FP32. Same base and
coverage best1792 delta b063f124e59c8a4c1b09c7bf2a8423846f73489d7ce8cd6649735745bc6cebf5.
New optimizer in both; not exact resume. Train all six encoder blocks/classifier
(10,647,169 parameters), embeddings/pooler frozen; total22,713,601 unchanged.

Select128 curriculum states via sorted groups shuffled with Python Random(17).
Use all eight questions/state (1,024 parents), shuffle parent order seed17, then
repeat that order four times. Original questions: first4,096 original IDs from
completed coverage run's ordered train_ids. Interleave4original+4curriculum.
Control uses original wording each encounter. Wording uses view0,1,2,3 by pass
for comparison/time tasks, view0 for count/conditional. Both have identical
parent order, targets, options, state, task/source counts and8192 encounters.
5,120 distinct parent questions; repeated parent encounters3,072. View diversity
is the sole intervention (both syntax and synonyms change jointly).

Validation remains original2540/curriculum2048 IDs/order. Evaluate0 and every128;
earliest minimum equal-source mean NLL includes0. Preserve best/final separately.
Do not stop for quality fluctuations; stop for numeric/freeze/data/storage errors.
No transfer diagnostic or sealed audit participates. Selected/endpoint trade-offs
and all failed attempts retained. No global95% or clean audit claim.

Separate two-update smoke per arm, separate outputs/processes, full random-state
reset for each main run. Verify exact best reload, initial validation parity to
coverage1792, source hashes, frozen weights and consumed-parent accounting.
Compare all scheduled learning curves; do not select a lucky seed. This bounded
experiment tests repeated wording views, not generic model capacity limits.
