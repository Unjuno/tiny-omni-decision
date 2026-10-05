from __future__ import annotations

from collections import OrderedDict


def install_video_decode_cache(
    processor: object, *, video_num_frames: int, capacity: int = 4
) -> None:
    """Reuse exact sampled frames for adjacent questions in one fixed-config evaluation."""
    if capacity < 1:
        raise ValueError("video decode cache capacity must be positive")
    if video_num_frames < 1:
        raise ValueError("video_num_frames must be positive")
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None:
        return
    original_fetch = video_processor.fetch_videos
    cache: OrderedDict[tuple[str, int], object] = OrderedDict()

    def cached_fetch(video: object, sample_indices_fn: object = None) -> object:
        if not isinstance(video, str):
            return original_fetch(video, sample_indices_fn=sample_indices_fn)
        key = (video, video_num_frames)
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        result = original_fetch(video, sample_indices_fn=sample_indices_fn)
        cache[key] = result
        if len(cache) > capacity:
            cache.popitem(last=False)
        return result

    video_processor.fetch_videos = cached_fetch
