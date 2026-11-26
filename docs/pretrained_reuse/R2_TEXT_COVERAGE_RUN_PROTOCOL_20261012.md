# Broader original-text coverage learning run

User-authorized local work; storage recheck now shows8.24GB free. Preserve every
previous run. One new seed17 run,2048 updates, eight questions/update, and no
within-run ID repeats. Use the exact sample-order file from coverage-preflight-01,
SHA f32de459ec5ddf7d5d0ec89c012d2de512099028ad61e9ce9d065697b23828bb.

Initialize the original22,713,601-parameter Cross-Encoder from deterministic
control selected step1664 (best-delta SHA0b785581bc5db5db01a9b39494af41b32d26fc6958d2c6191ccd74edcf016882).
No numeric embedding delta. Update all six encoder blocks/classifier10,647,169
parameters; embeddings/pooler remain frozen. New AdamW2e-5,weight decay.01,
clip1,constant LR,dropout off,deterministic algorithms,TF32 off,cuBLAS4096:8.
This is optimizer-reset continuation, not exact resume or fresh-base training.

Same original2540 and curriculum2048 validation IDs/order/preprocessing512 cap,
question/candidate first and truncate state only. Verify step0 metrics against
control1664 before updates. Earliest minimum equal-source meanNLL including
step0 selects best. Evaluate every128 updates, retain best/final separately,
all predictions and per-type metrics. No early stopping based on quality;
nonfinite/frozen-weight or disk-integrity failures stop safely.

The16384-question schedule uses8192 original questions absent from previous
all-block runs and8192 reused curriculum questions. All original records were
used in inherited final-block-only training. Claims of lifetime-new data are
forbidden. This run changes data exposure and adds optimizer updates; without
a matched repeated-data continuation it cannot isolate a causal coverage effect.

Smoke in a separate process/output on16 train and8+8 validation questions first.
Record full-run sourcecommit/environment/hash, frozen-state integrity, initial
validation parity, consumed/unique/state/type counts,gradientnorm,walltime,VRAM,
exact selected checkpoint reload. Do not re-evaluate any sealed audit or transfer
set. Do not promote the result automatically or claim general95% from subgroups.
Storage:160MiB output plus512MiB reserve; monitor during checkpointing.
