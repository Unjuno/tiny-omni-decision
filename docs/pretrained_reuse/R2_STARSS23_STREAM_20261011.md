# STARSS23 sequential-input parity

`scripts/starss23_stream.py` supplies audio/video iterators for the fixed
five-second windows. The audio reader opens each WAV once and seeks to each
requested sample interval. The video reader opens and decodes each recording
once, yielding eight frames at the same bin-midpoint presentation timestamps
as the original smoke. Only one window's selected frame list is accumulated;
consumers must release each yielded array rather than collect all windows.
This prevents storage of a whole recording's decoded RGB frames. Peak memory
reduction has not been benchmarked.

The reader rejects empty, overlapping, out-of-order or non-five-second windows,
wrong audio formats, incomplete media, missing/nonmonotonic video timestamps,
unsupported timeline offsets and selected frames beyond the 50 ms tolerance.
It performs no model inference, training, label filtering or split assignment.
Callers must verify source hashes and frozen split membership before using it.

## Real parity check

On 2026-10-11, the same two dev-train recordings and four intervals from
`starss23-window-smoke-01` were read through both iterators. Source media and
saved-window SHA-256 values were verified first. For each interval,
`numpy.array_equal` compared all audio samples, RGB pixels and timestamps to
the saved `.npz` arrays. All twelve comparisons passed exactly. This also
preserves the inputs used by the prior frozen-feature probe; features were
not re-trained or re-evaluated here.

The check took 2.337 seconds, including local source/archive hashing and
comparison; it is not an encoder latency measurement. The small result is
`R2_STARSS23_STREAM_20261011.json`. External result directory:
`C:\CodexArtifacts\pretrained-reuse-r2\starss23-stream-parity-01`.
The exact reader hash is saved in the result. Environment: existing Windows
Python3.11.9, NumPy2.4.6, PyAV18.1.0; CPU-only, no dependency changes.

Equivalent parity procedure for each source pair:

```python
windows = [(0, 5), (5, 10)]
for reference, audio, (video, timestamps) in zip(
    saved_window_rows,
    audio_windows(source_audio, windows),
    video_windows(source_video, windows),
    strict=True,
):
    with np.load(reference["path"], allow_pickle=False) as saved:
        assert np.array_equal(audio, saved["audio"])
        assert np.array_equal(video, saved["video"])
        assert np.array_equal(timestamps, saved["video_seconds"])
```

The 78-recording acquisition is a separate live process. This parity result
does not certify its completion or provide any independent quality evidence.
Next integrate the iterators with one loaded frozen encoder per extraction
process, after the complete acquisition and window-manifest gates pass.
