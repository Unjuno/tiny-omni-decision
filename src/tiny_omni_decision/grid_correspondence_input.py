"""Separate public audio/video correspondence inputs from private GRID labels."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class CorrespondenceInput:
    audio: bytes
    videos: tuple[bytes, bytes, bytes, bytes]
    option_ids: tuple[str, str, str, str]


def build_input(row: dict, audio_by_code: Mapping[str, bytes],
                video_by_code: Mapping[str, bytes], query_index: int):
    """Return model input and private target separately, rejecting ambiguity."""
    codes = row.get('codes')
    ids = row.get('option_ids')
    queries = row.get('queries')
    if (not isinstance(codes, list) or len(codes) != 4
            or any(not isinstance(c, str) or not c for c in codes) or len(set(codes)) != 4
            or not isinstance(ids, list) or len(ids) != 4
            or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != 4
            or not isinstance(queries, list) or len(queries) != 4
            or not isinstance(query_index, int) or isinstance(query_index, bool)
            or not 0 <= query_index < 4):
        raise ValueError('invalid quartet structure')
    expected = {(c, i) for c, i in zip(codes, ids, strict=True)}
    if any(not isinstance(q, dict) or not isinstance(q.get('audio_code'), str)
           or not isinstance(q.get('target_id'), str) for q in queries):
        raise ValueError('invalid query mapping')
    pairs = [(q.get('audio_code'), q.get('target_id')) for q in queries]
    if len(set(pairs)) != 4 or set(pairs) != expected:
        raise ValueError('query/target mapping differs from candidates')
    if set(audio_by_code) != set(codes) or set(video_by_code) != set(codes):
        raise ValueError('missing or unexpected media')
    if any(not isinstance(v, bytes) or not v for v in (*audio_by_code.values(),
                                                        *video_by_code.values())):
        raise ValueError('media must be nonempty bytes')
    videos = tuple(video_by_code[c] for c in codes)
    if len({hashlib.sha256(v).digest() for v in videos}) != 4:
        raise ValueError('duplicate video bytes')
    query = queries[query_index]
    public = CorrespondenceInput(audio_by_code[query['audio_code']], videos, tuple(ids))
    return public, query['target_id']
