"""Inventory all ADL activities with bounded, CRC-checked source ranges."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from scripts.audit_adl_participant_split import PROTOCOL_SHA, protocol_digest, summarize_activity
from scripts.audit_adl_zip_member import (
    MAX_COMPRESSED,
    MAX_UNCOMPRESSED,
    RECORD_SHA,
    central_records,
    decode_local_member,
    exact_range,
)
from scripts.audit_robomnist_archive_directories import parse_eocd

INVENTORY_SHA = 'efa745b87a0d51066a921366aef67cc21c53ab7283cc271d4ea1802fff34f3ba'
TWO_RECEIPT_SHA = 'a4ee1e4be98f458fc46512c30b8a5f1b0053a8be8ed07d653f45543343a8d27a'
THIRD_RECEIPT_SHA = 'e8af54b28ca7232a74cc3a86ce7427405d7509cb80a6f9bf6f526d251b9ce3bc'


def select_jobs(files):
    archives = [item for item in files if item.get('key', '').endswith('.zip')]
    numbers = []
    for item in archives:
        match = re.fullmatch(r'(\d{2})_[A-Za-z0-9_-]+\.zip', item['key'])
        if (not match or not isinstance(item.get('size'), int) or item['size'] <= 65536
                or not re.fullmatch(r'md5:[0-9a-f]{32}', item.get('checksum', ''))
                or urlparse(item.get('links', {}).get('self', '')).scheme != 'https'):
            raise ValueError('invalid source archive metadata')
        numbers.append(int(match.group(1)))
    if len(archives) != 25 or set(numbers) != set(range(25)):
        raise ValueError('expected exactly one archive for each of 25 activities')
    return sorted(archives, key=lambda item: (item['size'], item['key']))


def summarize_inventory(rows, expected=25):
    passes = [row for row in rows if row.get('status') in {'PASS_MEMBER_CRC', 'PASS_FULL_MD5'}]
    unsupported = len(rows) - len(passes)
    missing_coverage = sum(not row['split']['coverage_pass'] for row in passes)
    unassigned = sum(row['split']['unassigned_events'] for row in passes)
    return {
        'expected_activities': expected,
        'recorded_activities': len(rows),
        'inventory_finished': len(rows) == expected,
        'unsupported_or_failed': unsupported,
        'classes_without_both_splits': missing_coverage,
        'unassigned_events': unassigned,
        'complete': len(rows) == expected and not unsupported and not missing_coverage
        and not unassigned,
    }


def verified_json(path, expected_hash):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError(f'prior receipt hash mismatch: {path}')
    return json.loads(raw)


def preexisting_results(two_path, third_path):
    two = verified_json(two_path, TWO_RECEIPT_SHA)
    third = verified_json(third_path, THIRD_RECEIPT_SHA)
    if (len(two['archives']) != 2 or two['protocol_sha256'] != PROTOCOL_SHA
            or third['participant_split_protocol_sha256'] != PROTOCOL_SHA
            or third['status'] != 'PASS_MEMBER_CRC_SOURCE_FEASIBILITY_ONLY'):
        raise ValueError('prior source receipt mismatch')
    rows = {row['archive']: {'archive': row['archive'], 'status': 'PASS_FULL_MD5',
                             'archive_sha256': row['archive_sha256'], 'split': row['summary'],
                             'whole_archive_md5_verified': True,
                             'five_sensors_checked': True}
            for row in two['archives']}
    if set(rows) != {'23_take-from-stove.zip', '10_light-off.zip'}:
        raise ValueError('unexpected fully verified activities')
    rows[third['archive']] = {
        'archive': third['archive'], 'status': 'PASS_MEMBER_CRC',
        'member_sha256': third['member_sha256'], 'split': third['split'],
        'whole_archive_md5_verified': False, 'five_sensors_checked': False,
    }
    if third['archive'] != '17_stop-toaster.zip':
        raise ValueError('unexpected partial activity')
    return rows


def inspect_partial(item, session):
    tail_start = item['size'] - 65536
    tail, tail_etag = exact_range(session, item['links']['self'], tail_start,
                                  item['size'] - 1, item['size'])
    _, entries, size, offset = parse_eocd(tail, tail_start, item['size'])
    if offset < tail_start or offset + size > item['size']:
        raise ValueError('central directory does not fit in bounded tail')
    records = central_records(tail[offset - tail_start:offset - tail_start + size], entries)
    selected = [row for row in records if row['name'].endswith('_multisensor-1.hdf5')]
    if len(selected) != 1:
        raise ValueError('expected one multisensor-1 member')
    member = selected[0]
    if member['compressed'] > MAX_COMPRESSED or member['uncompressed'] > MAX_UNCOMPRESSED:
        return {'archive': item['key'], 'status': 'UNSUPPORTED_SIZE',
                'compressed_bytes': member['compressed'], 'decoded_bytes': member['uncompressed'],
                'whole_archive_md5_verified': False, 'five_sensors_checked': False}
    following = [r['offset'] for r in records if r['offset'] > member['offset']]
    boundary = min([offset, *following])
    end = min(boundary - 1, member['offset'] + 30 + len(member['name_bytes'])
              + 65535 + member['compressed'] - 1)
    block, member_etag = exact_range(session, item['links']['self'], member['offset'],
                                     end, item['size'])
    payload = decode_local_member(block, member)
    import h5py

    with h5py.File(io.BytesIO(payload), 'r') as h:
        participants, trials, times = h['participant'][:], h['trial'][:], h['timestamp'][:]
        rates = h['vibration_sample_rate'][:]
    if (len(times) != len(participants) or len(rates) != len(participants)
            or not np.isfinite(times).all() or not np.isfinite(rates).all()
            or np.any(rates <= 0)):
        raise ValueError('invalid participant/time/rate metadata')
    return {
        'archive': item['key'], 'status': 'PASS_MEMBER_CRC',
        'publisher_archive_md5': item['checksum'], 'tail_sha256': hashlib.sha256(tail).hexdigest(),
        'tail_bytes': len(tail), 'range_bytes': len(block), 'tail_etag': tail_etag,
        'member_etag': member_etag, 'member_name': member['name'],
        'member_sha256': hashlib.sha256(payload).hexdigest(),
        'member_crc32': f"{member['crc']:08x}", 'member_bytes': len(payload),
        'split': summarize_activity(participants, trials),
        'whole_archive_md5_verified': False, 'five_sensors_checked': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--two-receipt', type=Path, required=True)
    parser.add_argument('--third-receipt', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.record.read_bytes()).hexdigest() != RECORD_SHA:
        raise ValueError('source record hash mismatch')
    if protocol_digest('docs/pretrained_reuse/R2_ADL_PARTICIPANT_SPLIT_PROTOCOL_20261027.md') != (
        PROTOCOL_SHA
    ) or protocol_digest('docs/pretrained_reuse/R2_ADL_PARTIAL_INVENTORY_PROTOCOL_20261027.md') != (
        INVENTORY_SHA
    ):
        raise ValueError('preregistered protocol hash mismatch')
    jobs = select_jobs(json.loads(args.record.read_bytes())['files'])
    prior = preexisting_results(args.two_receipt, args.third_receipt)
    import psutil
    import requests

    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        'source_record_sha256': RECORD_SHA, 'split_protocol_sha256': PROTOCOL_SHA,
        'inventory_protocol_sha256': INVENTORY_SHA, 'prior_two_receipt_sha256': TWO_RECEIPT_SHA,
        'prior_third_receipt_sha256': THIRD_RECEIPT_SHA,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'ordered_archives': [job['key'] for job in jobs],
        'whole_archive_md5_verified_for_new_ranges': False,
        'scope': 'source feasibility only; no fitting or sealed audit',
    }
    (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    rows, started = [], time.perf_counter()
    try:
        with requests.Session() as session:
            for job in jobs:
                if psutil.virtual_memory().available < 2 * 2**30:
                    raise OSError('less than 2 GiB host memory available')
                if shutil.disk_usage(args.output_dir).free < 2**30:
                    raise OSError('less than 1 GiB disk free')
                row = prior.get(job['key'])
                if row is None:
                    row = inspect_partial(job, session)
                rows.append(row)
                target = args.output_dir / f"{job['key'][:2]}-summary.json"
                with target.open('x', encoding='utf8', newline='\n') as f:
                    json.dump(row, f, indent=2)
                    f.write('\n')
                print(json.dumps({'completed': len(rows), 'total': len(jobs),
                                  'archive': job['key'], 'status': row['status']}), flush=True)
        result = {**protocol, 'status': 'FINISHED_FEASIBILITY_ONLY',
                  'elapsed_seconds': time.perf_counter() - started,
                  'summary': summarize_inventory(rows), 'activities': rows,
                  'five_sensor_alignment_fully_checked_for_all': False,
                  'source_admitted_for_training': False}
        (args.output_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'status': result['status'], 'summary': result['summary']}), flush=True)
    except Exception:
        (args.output_dir / 'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
