# R2 conditional AV readout: synthetic CLEVRER development result

## Result

The approved five-arm comparison completed on the fixed synthetic CLEVRER AV
timing corpus. Every arm ran all 2,048 optimizer updates. **The conditional
audio/video readout did not improve this task: all selected checkpoints are at
chance, with approximately 0.6931 NLL. Do not adopt or promote this readout.**

This is a development-only result on generated 880 Hz tone energy paired with
CLEVRER video features. It is not evidence about natural audio/video, broad
multimodal reasoning, or the 95–98% product goal. The development validation
was already used by the earlier matched LSTM experiment and is not blind.
No sealed audit was opened.

## Frozen comparison

The five arms share seed 17, the same frozen training and validation IDs, the
same sampler order, batch size 64, AdamW learning rate 0.001, weight decay
0.0001, gradient clip 1.0, 2,048 updates, and validation every 128 updates.
Each arm consumed 130,512 sample exposures and all 3,696 unique training rows.
The dataset contains 773 train scenes and 240 validation scenes with zero
scene/media overlap. The fixed validation has 1,110 rows across 240 scenes.

| Arm | Exchange | Conditional gate | Trainable readout parameters | Selected step | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 native Attention | No residual | No | 889,345 | 128 | 50.00% | 0.6931472 | 0.5000000 | 0.000173 |
| Residual, no exchange / no gate | No | No | 1,732,420 | 128 | 50.00% | 0.6931472 | 0.5000000 | 0.000045 |
| Residual, no exchange / gate | No | Yes | 1,732,420 | 1,152 | 50.09% | 0.6931472 | 0.5000000 | 0.000900 |
| Residual, exchange / no gate | Yes | No | 1,732,420 | 128 | 50.00% | 0.6931472 | 0.5000000 | 0.000031 |
| Residual, exchange / gate | Yes | Yes | 1,732,420 | 128 | 50.00% | 0.6931472 | 0.5000000 | 0.000082 |

The paired bootstrap resampled 240 validation scenes 5,000 times (seed 17).
Accuracy differences from C0 were zero in three arms; the gated/no-exchange
arm was -0.10 percentage points, with a 95% interval from -0.31 to 0.00
points. NLL/Brier differences were only floating-point-scale changes and have
no practical meaning. This comparison provides no evidence that exchange or
gating helps.

The earlier LSTM comparison on these same frozen IDs selected joint AV at
83.51% / 0.3534 NLL and audio-only at 75.14% / 0.4577. Those are separate
development results, not a blind benchmark. The large gap indicates that this
candidate-query Attention implementation or its feature adapter failed to
learn the task under the matched budget; it does not establish a general limit
of attention or of conditional gating.

## Input and implementation checks

- Training/validation manifests, ordered IDs, frozen feature cache, and split
  boundaries were loaded through the existing hash-checked v2 loader.
- The MiniLM model for the two option queries came from the pinned revision
  `4ca70771034acceecb2e72475f72050fcdde4ddc`; its checkpoint SHA-256 matched
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- Synthetic audio is one normalized energy value in channel 0 of a zero-padded
  384-wide token. Video and audio share 16 normalized temporal positions. This
  adapter is a plausible source of information loss and needs an explicit
  scalar-audio projection comparison before drawing architecture conclusions.
- As an input sanity check after training, the audio-energy peak bin alone had
  correlation 0.56 with the label in train and validation. A threshold selected
  on train (`peak bin >= 6`) scored 75.62% train and 75.32% validation. This
  descriptive probe was not used for checkpoint or architecture selection; it
  shows that the frozen synthetic audio feature carries label information.
- The exact same initial C0 scores were reproduced by each residual arm at
  initialization. The 1-update CPU smoke exercised all five arms and best
  checkpoint reloads before the full run.
- The encoders stayed frozen and were not loaded during training. The readout
  plus the project’s 195,365,529-parameter frozen encoder ledger totals
  196,254,874 parameters for C0 and 197,097,949 for each AV arm, below 200M
  before quantization.

## Compute and artifacts

The full run used the local NVIDIA RTX 3080 Laptop GPU (16,384 MiB), started
through `scripts/run_with_hotgpu_guard.py`. The guard waited while PID 12596
occupied the GPU, then reserved 3,200 MiB and launched this run. PyTorch peak
allocated memory was 375,800,832 bytes (about 358 MiB). Candidate training took
169.61 seconds total and evaluation took 28.62 seconds. Cloud compute and cost:
zero. Run wall time was about 3 minutes 27 seconds after the child started.

Run artifacts are external to Git and immutable:

- `C:\CodexArtifacts\pretrained-reuse-r2\conditional-clevrer-av-readout-attempt-01\`
- Includes effective config, run metadata, all five learning histories,
  separate best/final checkpoints, best/final prediction JSONL files, and the
  result/paired-bootstrap summary.
- Result SHA-256: `dfe7a4fb9a15ad265ca4354660119a3d0cd7dedb5c5a177f577826c330e05895`.
- Effective config SHA-256: `655cdf1056e1df7639fcbac6e8153fa569091b70036ca0ebaaa87ff054c43512`.
- Ordered train/validation IDs: `977f1388e64c4d063e1496bed3747ee3e0c25639b1c069f4f889505b68686f9e` /
  `e21ec5bee231100f5949c38685890e452a140f6aade9f1bfd23efabaafd2c997`.
- Frozen feature cache SHA-256: `0b146fa637049d84ad88f0a71638cf30f6e967bd66add13ead0ab062bcbebdff`.
- Sampler-order SHA-256: `74724968a38753e3358d77b34a88108706c9e547d48c716cabb375863d607b1f`.

The 1-update smoke is separately preserved under
`C:\CodexArtifacts\pretrained-reuse-r2\conditional-clevrer-av-readout-smoke-01\`.
The newly downloaded MiniLM snapshot is separately preserved under
`C:\CodexArtifacts\pretrained-reuse-r2\conditional-av-minilm-pinned-cache-01\`.
Teacher, QAT, older v1/v2 run directories, frozen features, and sealed audit
were not modified.

## Next diagnostic

Do not extend training or tune against this validation result. First compare
the 384-wide zero-padding audio adapter with a dedicated learned scalar-to-192
projection, using the same C0/LSTM controls and a small predetermined smoke.
Then inspect per-stream gradients and candidate-score deltas. Any quality run
must keep the same source IDs and explicitly remain development-only; natural
AV claims require a separate rights-cleared, synchronized dataset with
task-grounded labels and source-grouped validation.
