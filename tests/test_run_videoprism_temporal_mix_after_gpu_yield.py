from scripts.run_videoprism_temporal_mix_after_gpu_yield import _create_handoff_guard


class _FakeAllocation:
    def __init__(self):
        self.values = []

    def fill_(self, value):
        self.values.append(value)


class _FakeCuda:
    def __init__(self):
        self.initialized = False
        self.synchronized = False

    def init(self):
        self.initialized = True

    def synchronize(self):
        self.synchronized = True


class _FakeTorch:
    uint8 = "uint8"

    def __init__(self):
        self.cuda = _FakeCuda()
        self.allocation = _FakeAllocation()

    def empty(self, size, *, dtype, device):
        assert size == 1
        assert dtype == self.uint8
        assert device == "cuda"
        return self.allocation


def test_handoff_guard_establishes_cuda_process_before_idle_wait():
    torch = _FakeTorch()

    guard = _create_handoff_guard(torch)

    assert guard is torch.allocation
    assert guard.values == [1]
    assert torch.cuda.initialized
    assert torch.cuda.synchronized
