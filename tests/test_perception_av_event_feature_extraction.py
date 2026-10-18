import importlib

import pytest


def test_target_index_tracks_the_source_after_deterministic_reordering():
    extraction = importlib.import_module("scripts.extract_perception_av_event_features")
    assert extraction.target_index_for_candidate_order([9, 2, 5], 5) == 2
    with pytest.raises(ValueError, match="source parent must occur exactly once"):
        extraction.target_index_for_candidate_order([9, 2, 5], 7)
