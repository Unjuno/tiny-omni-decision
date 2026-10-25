import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.extract_synthetic_av_readout_features import (
    audio_segment_hash,
    merge_feature_caches,
    require_cuda_gate,
    validate_audio_features,
    validate_rendered_manifest,
    validate_source_status,
    validate_video_features,
)
from scripts.prepare_synthetic_av_readout import render_assets
from tiny_omni_decision.synthetic_av_task import build_manifest


def test_cuda_feature_extraction_requires_hotgpu_reservation_without_cuda_init():
    with pytest.raises(RuntimeError, match="HotUGPU"):
        require_cuda_gate("cuda", {})
    require_cuda_gate("cpu", {})
    require_cuda_gate("cuda", {"TINY_OMNI_GPU_GATE_RESERVED_MIB": "1024"})


def test_video_extractor_import_does_not_require_renderer_only_pillow():
    code = """
import importlib.abc, sys
class BlockPillow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'PIL' or fullname.startswith('PIL.'):
            raise ModuleNotFoundError('renderer-only Pillow blocked for this test')
sys.meta_path.insert(0, BlockPillow())
import scripts.extract_synthetic_av_readout_features
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr


def test_video_feature_cache_requires_eight_ordered_finite_vectors():
    ids = ["seq-a"]
    values = np.zeros((1, 8, 768), dtype=np.float32)
    validate_video_features(ids, values)
    with pytest.raises(ValueError, match="shape"):
        validate_video_features(ids, np.zeros((1, 7, 768), dtype=np.float32))
    values[0, 3, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_video_features(ids, values)


def test_audio_features_require_four_finite_tone_vectors():
    validate_audio_features([0, 1, 2, 3], np.zeros((4, 384), dtype=np.float32))
    with pytest.raises(ValueError, match="symbol IDs"):
        validate_audio_features([0, 1, 2, 4], np.zeros((4, 384), dtype=np.float32))


def test_tone_segment_hash_is_stable_and_content_sensitive():
    left = np.arange(64, dtype=np.int16)
    right = left.copy()
    assert audio_segment_hash(left) == audio_segment_hash(right)
    right[-1] += 1
    assert audio_segment_hash(left) != audio_segment_hash(right)


def test_pinned_source_check_allows_only_known_install_generated_files():
    validate_source_status("?? build/\n?? videoprism.egg-info/\n?? videoprism/__pycache__/")
    with pytest.raises(ValueError, match="source checkout has unexpected changes"):
        validate_source_status(" M videoprism/models.py")
    with pytest.raises(ValueError, match="source checkout has unexpected changes"):
        validate_source_status("?? modified-source/")


def test_rendered_manifest_gate_checks_file_hash_and_split_identity(tmp_path: Path):
    manifest = render_assets(
        build_manifest(train_groups=1, validation_groups=1, seed=17), tmp_path / "assets"
    )
    validate_rendered_manifest(manifest, tmp_path / "assets")
    first = next(iter(manifest["assets"]["videos"].values()))
    media_path = tmp_path / "assets" / first["path"]
    media_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_rendered_manifest(manifest, tmp_path / "assets")


def test_feature_merge_preserves_complete_groups_and_eight_step_alignment():
    manifest = build_manifest(train_groups=2, validation_groups=2, seed=17)
    groups = [manifest["groups"][0], manifest["groups"][2]]
    sequence_ids = list(
        dict.fromkeys(sequence for group in groups for sequence in group["sequence_ids"].values())
    )
    video = np.zeros((len(sequence_ids), 8, 768), dtype=np.float32)
    audio = np.zeros((4, 384), dtype=np.float32)
    merged = merge_feature_caches(manifest, sequence_ids, video, [0, 1, 2, 3], audio)
    assert merged["video_sequence_indices"].shape == (16,)
    assert merged["audio_symbol_indices"].shape == (16, 8)
    assert set(merged["group_ids"]) == {group["group_id"] for group in groups}
    with pytest.raises(ValueError, match="complete groups"):
        merge_feature_caches(manifest, sequence_ids[:1], video[:1], [0, 1, 2, 3], audio)
