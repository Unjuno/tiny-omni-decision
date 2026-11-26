# MOMENT-1-small sensor checkpoint audit

Date: 2026-10-11. Scope: pinned checkpoint/source provenance, model load, shape, parameter accounting, and one local CUDA embedding smoke. No model training, task fitting, checkpoint modification, or UCI HAR test-split access occurred.

## Checkpoint and source

- Model: `AutonLab/MOMENT-1-small`, Hub revision `411e288267f82cce86296dbe4d6c8bc533cc162f`.
- Source: `moment-timeseries-foundation-model/moment`, commit `38f7310ad594100747ca2a8357e9c7ca7d323e0e`, package version 0.1.5, MIT source license.
- `model.safetensors`: 151,615,328 bytes, SHA-256 `785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877`. The 80 tensors contain 37,901,512 float32 parameters and 151,606,048 tensor bytes. The embedding task loads 35,337,408 parameters; the card's approximate 37.9M total includes additional task/head parameters.
- `config.json` SHA-256: `6456b8e6f017323e8ef372c338c993380e246cb6291ffb8ca21fe7dfc53d0a6f`.
- The model card declares MIT. It also warns that only the reconstruction head is pretrained and classification/forecasting heads require fine-tuning. This audit uses the documented embedding task and makes no classification-quality claim.

## Local runtime smoke

Using the pinned official source directly through `PYTHONPATH` (no install or upgrade), the embedding pipeline loaded the local, hash-verified checkpoint and processed a zero tensor with shape `[1, 6, 128]`, matching UCI HAR's six-channel, 128-sample window shape. `reduction="none"` returned `[1, 6, 16, 512]`, all finite.

- GPU: local RTX 3080 Laptop.
- Forward: 0.223 s for this single smoke call; peak allocated 162,730,496 bytes, peak reserved 184,549,376 bytes.
- Environment: Windows x86-64, Python 3.11.9, PyTorch 2.6.0+cu124, CUDA runtime 12.4, Transformers 5.6.2, NumPy 2.4.6, safetensors 0.8.0.
- The output is an embedding only. No candidate scorer, calibration, task accuracy, or option probabilities were evaluated.

The external result record is `C:\CodexArtifacts\pretrained-reuse-r1\moment-small-sensor-r1-attempt-01\r1-sensor-audit-result.json`, SHA-256 `ddc9db9da322f188d37e53fa0c7035b9b3de97e6b5631fb2f6795cf4f94cb578`. Checkpoint, source checkout, input archive, scripts, and result remain outside Git.

## UCI HAR overlap gate

The UCI Smartphone-Based Recognition of Human Activities and Postural Transitions dataset (UCI dataset 341) is listed by UCI as CC BY 4.0 and contains accelerometer and gyroscope readings sampled at 50 Hz. Its official archive was downloaded to the external artifact directory (79,596,192 bytes; SHA-256 `4ac4ae064227c07045a99551876b54d204837e995e298b6488559e209b3deb09`). Only archive member names and `Train/subject_id_train.txt` were read to verify archive shape; no `Test/` member contents were read or extracted. The train subject list contains 21 distinct IDs.

The MOMENT paper and official Time-series PILE card say pretraining includes the UCR/UEA classification archive (159 datasets), and the paper says pretraining uses each dataset's training split. The public source material inspected here does not provide a complete constituent-name list that proves UCI HAR is absent. Whether this specific UCI HAR corpus overlaps MOMENT pretraining is therefore **UNKNOWN**. To prevent a potentially contaminated result from influencing selection, no UCI HAR quality evaluation was run and this corpus must not be used to select MOMENT-1-small unless the overlap is resolved or the evaluation is explicitly treated as contaminated.

## Decision

**Runtime smoke: PASS. Sensor task quality: NOT MEASURED. Release eligibility: NOT ESTABLISHED.** MOMENT-1-small is a technically viable frozen sensor-feature candidate for further provenance review. It does not by itself implement calibrated Decision option probabilities. R0 still lacks a selected deployment device and task-specific quality/confidence gates; the user's provisional 100–200M parameter and <=200 MB package figures remain engineering targets only.

Primary sources: [MOMENT-1-small model card](https://huggingface.co/AutonLab/MOMENT-1-small/tree/411e288267f82cce86296dbe4d6c8bc533cc162f), [official MOMENT source commit](https://github.com/moment-timeseries-foundation-model/moment/tree/38f7310ad594100747ca2a8357e9c7ca7d323e0e), [Time-series PILE card](https://huggingface.co/datasets/AutonLab/Timeseries-PILE), [MOMENT paper](https://arxiv.org/abs/2402.03885), and [UCI HAR dataset page](https://archive.ics.uci.edu/dataset/341/smartphone%2Bbased%2Brecognition%2Bof%2Bhuman%2Bactivities%2Band%2Bpostura).
