from types import SimpleNamespace

from scripts.extract_perception_temporal_match_efficientat import (
    cuda_memory_metrics,
    device_name,
)


class _ForbiddenCuda:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"CPU execution must not access torch.cuda.{name}")


def test_cpu_feature_extraction_metadata_never_queries_cuda():
    torch = SimpleNamespace(cuda=_ForbiddenCuda())
    device = SimpleNamespace(type="cpu")

    assert device_name(torch, device) == "cpu"
    assert cuda_memory_metrics(torch, device) == (None, None)


def test_cuda_feature_extraction_records_device_and_memory():
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_name=lambda device: "Test GPU",
            max_memory_allocated=lambda device: 123,
            max_memory_reserved=lambda device: 456,
        )
    )
    device = SimpleNamespace(type="cuda")

    assert device_name(torch, device) == "Test GPU"
    assert cuda_memory_metrics(torch, device) == (123, 456)
