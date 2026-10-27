import json

import pytest

pytest.importorskip('torch')
pytest.importorskip('av')
pytest.importorskip('transformers')
pytest.importorskip('safetensors')
from scripts.smoke_grid_cross_source_json import validate_json


def test_validate_probability_distribution():
    ids = ['a', 'b', 'c', 'd']
    good = {'selected_option_id': 'a', 'probabilities': dict.fromkeys(ids, 0.25)}
    assert validate_json(json.dumps(good), ids) == good
    good['probabilities']['a'] = 0.5
    with pytest.raises(ValueError, match='probabilities'):
        validate_json(json.dumps(good), ids)
