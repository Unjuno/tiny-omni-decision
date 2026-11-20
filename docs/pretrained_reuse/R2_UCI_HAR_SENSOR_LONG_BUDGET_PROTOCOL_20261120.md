# UCI HAR sensor-only long-budget follow-up — frozen protocol

## Rationale

The first fixed 12-epoch comparison selected epoch 12 for both models. At that
endpoint, training and validation NLL were still falling for the static and
LSTM candidates. The separate one-time public-test comparison is already
closed and will not be repeated. It showed a substantial LSTM Accuracy loss,
so the longer run is justified only by the still-improving validation curves;
test values are excluded from all further selection and analysis choices.

## Frozen follow-up

- New output: `C:\CodexArtifacts\pretrained-reuse-r2\uci-har-sensor-v1\attempt-02-long100`.
- Start both models from their deterministic seed-17 random initialization;
  do not warm-start from the 12-epoch checkpoints or claim optimizer resume.
- Keep the exact UCI HAR outer/nested archive hashes, train/validation windows
  and order, participant IDs, six channels, train-only normalization, static
  and LSTM architectures, AdamW settings, batch size, CPU execution, and
  candidate selector from the first run.
- The only planned experiment change is total training budget: 100 epochs for
  both models, with the same fixed learning rate. No early stop; preserve each
  epoch's metrics, the best validation-NLL checkpoint, and the terminal
  checkpoint.
- With 6,028 train windows and batch size 128, the fixed sampler processes
  6,028 examples / 48 optimizer updates per epoch, or 602,800 example
  presentations / 4,800 updates per model across 100 epochs. Unique training
  examples remain 6,028 per model. The first LSTM run measured 282.50 seconds
  for 12 epochs; this projects roughly 40 minutes total for both 100-epoch
  arms on CPU, before the long run's measured time is available.
- This code path reads only the official train split. Do not invoke `--phase
  audit`; the UCI public test was already read once after the first candidate
  freeze and is permanently closed to this experiment.
- Compare the first 12 epoch metrics to the original run as an execution
  determinism check. Use all later validation metrics only to describe the
  learning curve and select each model's best epoch. Do not change data,
  architecture, sampling, seed, LR, or selector in response to results.

This is a validation-only budget follow-up for one narrow sensor activity
task. It does not qualify a product model, close the sensor deterministic-rule
baseline, establish a four-modality result, or justify any inference about a
general architecture ceiling.
