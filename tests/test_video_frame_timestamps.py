import numpy as np
import pytest

from tiny_omni_decision.video_frame_timestamps import decode_endpoint_frames

av = pytest.importorskip("av")


def test_real_decoder_retains_frame_timestamps(tmp_path):
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(10):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), index * 20, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    frames, metadata = decode_endpoint_frames(path)
    assert frames.shape == (8, 32, 32, 3)
    assert metadata["decoded_frame_count"] == 10
    assert metadata["indices"] == [0, 1, 3, 4, 5, 6, 8, 9]
    np.testing.assert_allclose(metadata["timestamps_ms"], [0, 100, 300, 400, 500, 600, 800, 900])
    assert frames[0].mean() < frames[-1].mean()
