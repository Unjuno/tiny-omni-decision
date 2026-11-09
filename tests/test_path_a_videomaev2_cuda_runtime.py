from __future__ import annotations

import pytest

from scripts.audit_path_a_videomaev2_cuda_runtime import run_cuda_smoke


def test_cuda_smoke_refuses_existing_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_cuda_smoke(output)


def test_cuda_smoke_requires_repeated_observations_before_model_load(tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        run_cuda_smoke(tmp_path / "new", repeats=1)

