# Compact video checkpoint follow-up

Question: can a smaller pretrained video tower leave enough of the <=200M
budget for stronger language reasoning? This is a source audit, not a benchmark
result or permission to replace the existing video tower on size alone.

- VideoMAE-small: pinned revision `1bf2b82c809609f5df1ea7648aee577e2e486b63`. The [Hub card](https://huggingface.co/MCG-NJU/videomae-small-finetuned-kinetics/tree/1bf2b82c809609f5df1ea7648aee577e2e486b63)
  labels it CC-BY-NC-4.0. The [upstream repository](https://github.com/MCG-NJU/VideoMAE)
  also states the main project is noncommercial. Excluded from the commercial
  product path, consistent with the earlier base-model audit. Not downloaded.
- X3D-S: [official model zoo](https://raw.githubusercontent.com/facebookresearch/pytorchvideo/f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0/docs/source/model_zoo.md) reports 3.79M
  parameters, 13x6 frame sampling and Kinetics classification benchmarks. This is
  not an exact local count or evidence of CLEVRER/conditional reasoning quality.
  The [repository license](https://raw.githubusercontent.com/facebookresearch/pytorchvideo/f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0/LICENSE) is Apache-2.0; the
  inspected README/license/model-zoo do not separately establish the downloaded
  checkpoint's terms. Keep REVIEW pending sufficient weight-license evidence,
  exact tensor count, preprocessing and actual finite forward. No weight download,
  runtime execution, environment installation, quality evaluation or adoption.

A 3.79M video tower would change the parameter allocation considerably, but an
activity-classification representation might still fail temporal relations.
Model names or parameter savings cannot satisfy the quality gate. Do not use
unknown rights or the new encoder to alter current frozen comparisons.

Pinned commit and source-document hashes are in the adjacent JSON. This check
used only official source pages/API metadata and did not open any final audit.
