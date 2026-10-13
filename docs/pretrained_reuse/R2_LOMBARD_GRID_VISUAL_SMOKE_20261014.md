# Real visual smoke: successful, CPU cost measured

Four full-release **training** clips passed frozen VideoPrism-B inference at its
native16-frame profile. Actual-PTS uniform endpoint sampling, official bilinear
resize/center crop288,RGB[0,1],FP32. No filename, transcript, gold word alignment
or model-quality scoring. Old8-frame CLEVRER/Teacher configs and caches are untouched.

Pinned source `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`, unmodified checkout;
checkpointSHA`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`;
114,365,184loaded parameters. All outputs `[1,4096,768]` finite and repeated
bitwise on the same CPU backend. Framewise spatial means `[16,768]` are saved,
preserving chronological positions and selected source PTS.

First compile/forward9.97s. Three warm forwards2.78/2.70/2.73s; smoke30.74s
including replays/preprocessing, excluding model load. Peak working set
4,313,677,824bytes. JAX0.6.2,TensorFlow2.18.1,Flax0.11.1,NumPy1.26.4,CPU.
5,367single forwards project to14,674.70s (~4.08h), excluding decode/preprocess/
load/replay. This is a **three-sample projection**, not an executed full job.
Repeat checks would add compute. No full visual extraction/training was launched.

## Actual runtime issue and available execution path

The previously used virtual-environment`python.exe` was rejected by Windows
application control before the script started. Its file/source/environment are
preserved. The permitted system Python3.11 successfully imported the existing
dependency directory in a separate process and ran the real smoke. Dependencies
were neither installed nor upgraded; the existing scientific package versions
remain separate from the normal Torch/NumPy process. This is an alternative
approved interpreter path, not a claim that the blocked launcher was repaired.
Reproduce with:

```powershell
python -c "import sys,runpy;sys.path[:0]=['C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv/Lib/site-packages','C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/source'];runpy.run_path('C:/CodexArtifacts/pretrained-reuse-r2/lombard_grid_videoprism_smoke.py',run_name='__main__')"
```

Evidence:`C:/CodexArtifacts/pretrained-reuse-r2/lombard-grid-videoprism-smoke-01/`
and the exact runner`.../lombard_grid_videoprism_smoke.py`; adjacent JSON includes
all inputs, PTS, timings, source hashes and runtime-entry scope. No paid compute,
sealed audit or existing checkpoint alteration. New Decision accuracy is unmeasured.

## Next action from the measurement

Audio full extraction is complete; visual CPU throughput is now a measured
inefficiency. Before committing to hours of extraction, inspect a bounded GPU
implementation of the **same** frozen VideoPrism weights/operators. Require exact
loaded parameter accounting, identical inputs/preprocessing and numerical parity
against the saved native CPU results, then measure runtime/VRAM. Backend conversion
must not silently change architecture, frame count, data or logits. If parity cannot
be established, keep the native runtime and report that constraint. This is an
execution optimization within Path A, not new backbone training/merging or proof
of better video quality. The full95–98% Omni objective remains active.
