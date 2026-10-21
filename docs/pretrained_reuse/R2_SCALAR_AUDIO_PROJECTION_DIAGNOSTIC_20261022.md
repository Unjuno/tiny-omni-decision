# R2 scalar audio projection diagnostic

## Result

The fixed-budget diagnostic completed for both configured arms (C0 and
exchange/gate), 2,048 optimizer updates each, with best-checkpoint reload and
selector checks passing. The learned scalar-to-192 projection did **not** make
the conditional AV readout learn this task: selected validation Accuracy was
50.00%, NLL 0.693147, Brier 0.500000 for both arms. Do not adopt this design
based on this experiment.

This is synthetic tone-energy timing paired with CLEVRER video features and a
previously observed development validation set. It is not evidence about
natural audio/video or broad multimodal decision quality. No sealed audit was
opened and no product artifact was changed.

## Frozen conditions and integrity

- Branch source: `codex/r2-conditional-av-readout-20261022`, commit
  `aaccca233e00564ae1f828ce0668bc96e0e3dee1`.
- Seed 17; C0 and exchange/gate; batch 64; 2,048 updates per arm; validation
  every 128 updates; AdamW LR 0.001; same fixed training/validation IDs and
  frozen feature cache as the preceding conditional AV attempt.
- Single intended architecture change: map the scalar synthetic audio value
  from channel 0 through a learned Linear(1,192) in the residual AV path.
  C0 retains its original zero-padded path. No data or encoder changes.
- Training rows: 3,696 (1,848 paired events); validation rows: 1,110
  (555 paired events). Unique training IDs consumed: 3,696; repeated sampling
  occurred because the run used batch 64 for 2,048 updates (130,512 example
  exposures per candidate). The split reports zero media-hash and scene
  overlap.
- Ordered train/validation ID hashes match the prior run: `977f1388e64c4d063e1496bed3747ee3e0c25639b1c069f4f889505b68686f9e` /
  `e21ec5bee231100f5949c38685890e452a140f6aade9f1bfd23efabaafd2c997`.
- Effective config SHA-256:
  `2694a926e94eb9d901e65c4ac4ac1f9eef9370f37328f0064c1a54913fd7c53`.
- Results SHA-256:
  `5489b5ab09be2cec995011917fd588ede8dd90939b2b312b0619fb4bf31c17ed`.
- Run metadata says complete; all best/final checkpoints and best/final
  per-example predictions were saved outside Git. Best checkpoint hashes:
  C0 `05d6871c8bb6382ffc7730c3b86807b9966472ce594d278935b7cd01bd72901f`,
  exchange/gate `bf770b01306e39b002e9aed8c988527278a51f379f21d414ec2d8fd7c7db3e7f`.

## Metrics

| Candidate | Selected step | Validation Acc. | NLL | Brier | ECE (15 bins) | Final train Acc. | Final validation Acc. | Trainable params | Peak allocated VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 native Attention | 128 | 50.00% | 0.6931472 | 0.500000 | 0.000173 | 50.00% | 50.00% | 889,345 | 328.4 MiB |
| exchange + gate + scalar projection | 128 | 50.00% | 0.6931472 | 0.500000 | 0.000024 | 50.00% | 50.00% | 1,732,804 | 358.4 MiB |

The exchange/gate arm briefly reached 50.27% validation Accuracy at step 1,920,
then ended at 50.00%; train Accuracy at step 1,920 was 50.11% and also ended
at 50.00%. The fixed selector retained step 128. This is unstable near-chance
movement, not sustained evidence of learning. A scene-clustered paired
bootstrap against C0 found zero Accuracy difference; NLL/Brier changes were
approximately `4.5e-8` and are numerically negligible.

The prior zero-padded exchange/gate arm also selected step 128 at 50.00%
Accuracy / 0.693147 NLL. Thus this diagnostic found no measurable gain from
the scalar adapter under the matched configuration. It does not show that
audio is uninformative: the separate train-threshold probe already found
signal in the synthetic audio feature. The failure could remain in optimization,
readout interaction, or task implementation; do not infer an architecture-wide
limit from this one comparison.

## Runtime and artifacts

The local RTX 3080 Laptop GPU was used through `scripts/run_with_hotgpu_guard.py`;
no cloud compute was used. The guard waited for cooperative GPU yield and only
then launched the child. Total recorded candidate training time was 56.06 s;
evaluation 9.22 s; peak allocated memory 358.4 MiB. Artifacts are at
`C:\CodexArtifacts\pretrained-reuse-r2\conditional-clevrer-av-scalar-audio-attempt-01\`
(`run-metadata.json`, effective config, histories, results, checkpoints and
predictions). Checkpoint binaries remain outside Git.

The run completed 2,048 updates per candidate, but its timestamps and reported
training durations should be treated as program telemetry rather than a wall
clock benchmark. The Windows PyTorch allocator warned that
`expandable_segments` is unsupported; it did not prevent completion.

## Scope and next step

Do not run more tuning against this exposed validation set. Preserve this as a
negative diagnostic. Further architecture conclusions require an independent,
rights-cleared synchronized AV task with source-grouped splits and a training
signal verified before model comparison. The prior conditional AV and scalar
diagnostic both remain research artifacts, not product Teachers. Existing v1,
v2, QAT, corpus, checkpoint and sealed-audit artifacts were not modified.
