"""Option-cache identities and records; no model, tensor or dataset loading.

The identity must be captured by the exporter, not inferred from filenames.
A valid identity does not prove the exporter actually ran the declared model.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any

LABELS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
IDENTITY_KEYS = {'purpose', 'sample_id', 'example_sha256', 'input_sha256', 'option_texts',
                 'option_labels', 'option_token_ids', 'target', 'media_sha256s', 'teacher'}
TEACHER_KEYS = {'master_sha256', 'model_revision', 'processor_revision', 'preprocessing_sha256'}
RECORD_KEYS = {'schema_version', 'kind', 'identity', 'temperature', 'teacher_option_logits',
               'cache_key'}


def canonical_json(value: Any) -> bytes:
    """Canonical UTF-8 JSON used for identities and JSONL rows (newline not included)."""
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def exact_keys(value: Any, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f'{label}: expected exactly {sorted(keys)}')


def require_digest(value: Any, size: int = 64) -> None:
    if not isinstance(value, str) or re.fullmatch(rf'[0-9a-f]{{{size}}}', value) is None:
        raise ValueError(f'expected lowercase {size}-character digest')


def validate_teacher_identity(teacher: dict[str, object]) -> None:
    exact_keys(teacher, TEACHER_KEYS, 'teacher')
    for key, value in teacher.items():
        require_digest(value, 40 if key.endswith('revision') else 64)


def validate_cache_identity(identity: dict[str, object]) -> None:
    exact_keys(identity, IDENTITY_KEYS, 'cache identity')
    if identity['purpose'] not in ('training', 'development'):
        raise ValueError('cache purpose must be training or development, never audit/test')
    if not isinstance(identity['sample_id'], str) or not identity['sample_id'].strip():
        raise ValueError('nonempty sample_id required')
    for key in ('example_sha256', 'input_sha256'):
        require_digest(identity[key])
    validate_teacher_identity(identity['teacher'])
    options = identity['option_texts']
    if (not isinstance(options, list) or not 2 <= len(options) <= 62
            or any(not isinstance(s, str) or not s.strip() for s in options)
            or len(set(options)) != len(options)):
        raise ValueError('require 2..62 distinct nonempty ordered option texts')
    if identity['option_labels'] != list(LABELS[:len(options)]):
        raise ValueError('option labels must retain the case-sensitive A-Z/a-z/0-9 contract')
    tokens = identity['option_token_ids']
    if (not isinstance(tokens, list) or len(tokens) != len(options)
            or any(type(t) is not int or t < 0 for t in tokens)
            or len(set(tokens)) != len(tokens)):
        raise ValueError('one distinct nonnegative token ID per ordered option required')
    if type(identity['target']) is not int or not 0 <= identity['target'] < len(options):
        raise ValueError('target must be a valid integer option index')
    media = identity['media_sha256s']
    if not isinstance(media, list):
        raise ValueError('ordered media hashes must be a list')
    for item in media:
        require_digest(item)


def make_teacher_cache_record(
    identity: dict[str, object], option_logits: list[float]
) -> dict[str, object]:
    """Bind raw option logits to the actual recorded input/teacher identity."""
    validate_cache_identity(identity)
    if not isinstance(option_logits, list) or len(option_logits) != len(identity['option_texts']):
        raise ValueError('one raw logit per option required; probabilities are not a substitute')
    try:
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in option_logits):
            raise ValueError('raw option logits must be finite numbers, not bool/string')
    except OverflowError as exc:
        raise ValueError('raw option logit exceeds numeric range') from exc
    key = hashlib.sha256(canonical_json(identity)).hexdigest()
    return {'schema_version': 1, 'kind': 'postquant_teacher_cache_record',
            'identity': copy.deepcopy(identity), 'temperature': 1.0,
            'teacher_option_logits': list(option_logits), 'cache_key': key}


def validate_teacher_cache_record(
    record: dict[str, object], expected: dict[str, object]
) -> None:
    """Reject stale targets, permutations, preprocessing, media or teacher revisions."""
    exact_keys(record, RECORD_KEYS, 'cache record')
    validate_cache_identity(expected)
    if type(record['schema_version']) is not int or record['schema_version'] != 1:
        raise ValueError('unsupported cache schema')
    if record['kind'] != 'postquant_teacher_cache_record':
        raise ValueError('unsupported cache record kind')
    if type(record['temperature']) not in (int, float) or record['temperature'] != 1.0:
        raise ValueError('this cache contract supports temperature 1.0 only')
    canonical = make_teacher_cache_record(record['identity'], record['teacher_option_logits'])
    if record['cache_key'] != canonical['cache_key']:
        raise ValueError('cache identity digest mismatch')
    if canonical_json(record['identity']) != canonical_json(expected):
        raise ValueError('cache record does not match the expected input/teacher identity')
