# One-factor lexical coverage control

Evidence: the selected numeric-range warm start improved unseen range interpolation
but regressed on wording (56.54% to42.09%). Test whether training wording coverage
helps, keeping numeric data, labels, item counts, optimizer and starting weights fixed.

Train-only corpus:8,192questions,1,024latent states. Seed101 assigns each state one
of four wording variants,2,048questions each; variant0 preserves the original.
Three new semantic-equivalent templates per task do not copy either observed
probe's normalized paraphrases. Existing state, numeric values, options, target,
IDs and row order are preserved. Corpus SHA256
`91c4b3f9ff81a3d01c5f4a9db0c83d548b33daab44a9ee879b236a5ded136645`.
No synthetic example duplication or new source licensing assumptions.

Initialize from parent narrow encoder-coverage selected2048
SHA4024788b1a9d17735612a98da6acb1a65af5d82a9d440b4fddb4ab62efb9d318,
not the later numeric-range model. New AdamW; not exact resume. Same seed17,
2,048updates,4original+4curriculum questions/update,2e-5constantLR,.01weight decay,
clip1,dropout off,all six existing encoder blocks/classifier trainable. Embeddings
and pooler frozen. Total22,713,601parameters;10,647,169trainable. No architecture
or inference size change. Same original data and validation IDs/order, earliest
minimum mean original/curriculum NLL selector including step0, every128updates.
Run the fixed budget unless resource/numeric/integrity safety fails. Separate
best/final, exact best reload and frozen hashes required. No audit or transfer
set participates in selection. Existing narrow selector can miss wording gains;
report that limitation rather than changing it after observing outcomes.

Separate-process2-update smoke first. New outputs lexical-range-smoke-01 and
conditional-repair-lexical-range-01 under C:/CodexArtifacts/pretrained-reuse-r2.
Check actual config/schedule against completed range-warmstart before interpreting
causally; only curriculum question text and its bookkeeping should differ.
No follow-up seed/budget extension under this protocol. LocalGPU only, approximately
20minutes, no paid compute. Preserve all prior corpora/checkpoints.

After completion: record all curves and original calibration regressions. Do not
rerun old transfer inputs or call training fit generalization. A separate fixed
unused-state/unused-wording evaluation must be specified before any new inference.
Successful training does not meet general text or multimodal95% targets.
