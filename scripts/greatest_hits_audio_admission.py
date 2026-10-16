"""Explicit admission of independently revalidated banks; producer failure stays intact."""

import copy
import json
from pathlib import Path

from scripts.native_video_cache import sha


def load_revalidated_audio(cache_root, verification_path, verification_sha256):
    root, path = Path(cache_root), Path(verification_path)
    if sha(path) != verification_sha256:
        raise ValueError('independent verification receipt hash differs')
    proof = json.loads(path.read_text(encoding='utf-8'))
    raw_path = root/'result.json'
    raw_sha = sha(raw_path)
    raw = json.loads(raw_path.read_text(encoding='utf-8'))
    if (proof['status'] != 'PASS_FULL_AUDIO_CACHE_REVALIDATION'
            or proof['source_commit'] != '933e5dc84cbf125b3a2965f4250fdd0de5ad3ef5'
            or proof['producer_status'] != 'INCOMPLETE'
            or Path(proof['source_cache']).resolve() != root.resolve()
            or proof['producer_result_sha256'] != raw_sha
            or raw['status'] != 'INCOMPLETE' or raw['phase'] != 'full'
            or raw['source_commit'] != '61530adc76935e4b5dbfbabf582ef1ec03cd1405'
            or raw['error'] != "ValueError('incomplete coverage or encoder state changed')"
            or raw['frozen_before'] != raw['frozen_after'] or raw['optimizer_updates'] != 0):
        raise ValueError('unsupported producer failure or unlinked verification')
    for key in ('events','parents','questions'):
        n = raw['expected_'+key]
        if (type(n) is not int or n <= 0 or raw['completed_'+key] != n
                or proof['verified_'+key] != n):
            raise ValueError('independent complete coverage is required')
    if (proof['input_replay_matches'] != raw['completed_events']
            or proof['feature_digest_matches'] != 2*raw['completed_events']
            or proof['smoke_feature_matches'] != 2*raw['matched_smoke_events']
            or proof['corrected_expected_reference_matches'] != raw['matched_old_audio_events']
            or proof['optimizer_updates'] != 0 or proof['encoder_replayed'] is not False
            or any(proof[k] is not False for k in ('quality_evaluated','sealed_audit_loaded',
                                                  'calibration_loaded'))):
        raise ValueError('independent cache verification checks are incomplete')
    expected = {str(raw_path):raw_sha}
    for name,digest in raw['artifacts'].items():
        if name != Path(name).name or sha(root/name) != digest:
            raise ValueError('saved artifact path/hash differs')
        expected[str(root/name)] = digest
    ledger_path = path.parent/'artifact-hashes.json'
    if (proof['artifacts'] != expected or sha(ledger_path) != proof['artifact_ledger_sha256']
            or json.loads(ledger_path.read_text(encoding='utf-8')) != expected
            or sha(raw_path) != raw_sha or sha(path) != verification_sha256):
        raise ValueError('independent artifact ledger or receipt changed')
    # This is a cache-valid view for binding, not a new producer run receipt.
    view = copy.deepcopy(raw)
    view['status'] = 'PASS_COMPLETE_AUDIO_QUERY_CACHE'
    view['audio_admission'] = dict(kind='independent-full-revalidation',
        producer_status='INCOMPLETE', producer_result_sha256=raw_sha,
        verification_sha256=verification_sha256, verification_path=str(path))
    return view
