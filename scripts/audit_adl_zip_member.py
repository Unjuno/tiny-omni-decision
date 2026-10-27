"""Inspect a bounded ZIP member from a CC BY source without saving media."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import time
import traceback
import zlib
from pathlib import Path

from scripts.audit_adl_participant_split import PROTOCOL_SHA, protocol_digest, summarize_activity
from scripts.audit_robomnist_archive_directories import parse_eocd

RECORD_SHA = 'c01bf60460981280c5063ea54c038d41c751bf3056220059c3896cbc5d2b97c6'
TAIL_SHA = '7a170fbc393f7ec17a61785443cf4a72e1e05e89d02a9df0bbe081f17acaa696'
MAX_COMPRESSED = 64 * 2**20
MAX_UNCOMPRESSED = 128 * 2**20


def central_records(data, expected_entries):
    records, offset = [], 0
    while offset < len(data):
        if offset + 46 > len(data) or data[offset:offset + 4] != b'PK\x01\x02':
            raise ValueError('invalid ZIP central directory')
        flags, method = struct.unpack_from('<HH', data, offset + 8)
        crc, compressed, uncompressed = struct.unpack_from('<III', data, offset + 16)
        name_len, extra_len, comment_len = struct.unpack_from('<HHH', data, offset + 28)
        local_offset = struct.unpack_from('<I', data, offset + 42)[0]
        end = offset + 46 + name_len + extra_len + comment_len
        if end > len(data) or 0xffffffff in (compressed, uncompressed, local_offset):
            raise ValueError('truncated or ZIP64 member')
        name_bytes = data[offset + 46:offset + 46 + name_len]
        name = name_bytes.decode('utf8' if flags & 0x800 else 'cp437')
        records.append({'name': name, 'name_bytes': name_bytes, 'flags': flags,
                        'method': method, 'crc': crc, 'compressed': compressed,
                        'uncompressed': uncompressed, 'offset': local_offset})
        offset = end
    if len(records) != expected_entries or len({r['name'] for r in records}) != len(records):
        raise ValueError('ZIP member count or uniqueness mismatch')
    return records


def decode_local_member(data, record):
    if len(data) < 30 or data[:4] != b'PK\x03\x04':
        raise ValueError('invalid local ZIP header')
    flags, method = struct.unpack_from('<HH', data, 6)
    local_crc, local_compressed, local_uncompressed = struct.unpack_from('<III', data, 14)
    name_len, extra_len = struct.unpack_from('<HH', data, 26)
    header_end = 30 + name_len + extra_len
    if header_end > len(data) or data[30:30 + name_len] != record['name_bytes']:
        raise ValueError('local ZIP name or header mismatch')
    if flags != record['flags'] or method != record['method'] or flags & 1:
        raise ValueError('local ZIP flags or method mismatch')
    if not flags & 8 and (local_crc, local_compressed, local_uncompressed) != (
        record['crc'], record['compressed'], record['uncompressed']
    ):
        raise ValueError('local ZIP size or CRC metadata mismatch')
    if record['compressed'] > MAX_COMPRESSED or record['uncompressed'] > MAX_UNCOMPRESSED:
        raise ValueError('member exceeds bounded size')
    compressed = data[header_end:header_end + record['compressed']]
    if len(compressed) != record['compressed']:
        raise ValueError('truncated ZIP payload')
    if method == 0:
        decoded = compressed
    elif method == 8:
        decomp = zlib.decompressobj(-15)
        decoded = decomp.decompress(compressed, record['uncompressed'] + 1)
        if not decomp.eof or decomp.unconsumed_tail or decomp.unused_data:
            raise ValueError('invalid deflate stream or size')
    else:
        raise ValueError('unsupported ZIP compression method')
    if len(decoded) != record['uncompressed'] or zlib.crc32(decoded) != record['crc']:
        raise ValueError('ZIP payload length or CRC mismatch')
    return decoded


def exact_range(session, url, start, end, total):
    expected = end - start + 1
    if expected <= 0 or expected > MAX_COMPRESSED + 131072:
        raise ValueError('range exceeds transfer cap')
    with session.get(url, headers={'Range': f'bytes={start}-{end}',
                                  'Accept-Encoding': 'identity'},
                     stream=True, timeout=(15, 30)) as response:
        if response.status_code != 206:
            raise ValueError(f'expected HTTP 206, got {response.status_code}')
        if response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
            raise ValueError('unexpected content range')
        if response.headers.get('Content-Length') != str(expected):
            raise ValueError('unexpected content length')
        data = response.raw.read(expected + 1)
        if len(data) != expected:
            raise ValueError('unexpected range body length')
        return data, response.headers.get('ETag')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-record', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    raw = args.source_record.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RECORD_SHA:
        raise ValueError('source record hash mismatch')
    item = sorted((r for r in json.loads(raw)['files'] if r['key'].endswith('.zip')),
                  key=lambda r: (r['size'], r['key']))[2]
    if item['key'] != '17_stop-toaster.zip' or item['size'] != 72770695:
        raise ValueError('preregistered source selection mismatch')
    protocol_path = Path('docs/pretrained_reuse/R2_ADL_PARTICIPANT_SPLIT_PROTOCOL_20261027.md')
    if protocol_digest(protocol_path) != PROTOCOL_SHA:
        raise ValueError('participant split protocol mismatch')
    import h5py
    import psutil
    import requests

    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        'source_record_sha256': RECORD_SHA, 'participant_split_protocol_sha256': PROTOCOL_SHA,
        'archive': item['key'], 'archive_publisher_md5': item['checksum'],
        'selection_rule': 'third smallest archive; its multisensor-1 HDF5 member',
        'expected_tail_sha256': TAIL_SHA, 'script_sha256': hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        'whole_archive_md5_verified': False,
        'use': 'source feasibility only, no fit, validation score or audit',
    }
    (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    tick = time.perf_counter()
    try:
        if psutil.virtual_memory().available < 2 * 2**30:
            raise OSError('less than 2 GiB host memory available')
        with requests.Session() as session:
            tail_start = item['size'] - 65536
            tail, tail_etag = exact_range(session, item['links']['self'], tail_start,
                                          item['size'] - 1, item['size'])
            if hashlib.sha256(tail).hexdigest() != TAIL_SHA:
                raise ValueError('ZIP tail differs from preflight')
            _, entries, size, offset = parse_eocd(tail, tail_start, item['size'])
            if offset < tail_start or offset + size > item['size']:
                raise ValueError('central directory outside fetched tail')
            records = central_records(tail[offset - tail_start:offset - tail_start + size], entries)
            selected = [r for r in records if r['name'].endswith('_multisensor-1.hdf5')]
            if len(selected) != 1:
                raise ValueError('expected exactly one multisensor-1 member')
            member = selected[0]
            if member['compressed'] > MAX_COMPRESSED or member['uncompressed'] > MAX_UNCOMPRESSED:
                raise ValueError('member exceeds bounded size')
            following = [r['offset'] for r in records if r['offset'] > member['offset']]
            boundary = min([offset, *following])
            end = min(boundary - 1, member['offset'] + 30 + len(member['name_bytes'])
                      + 65535 + member['compressed'] - 1)
            if psutil.virtual_memory().available < 2 * 2**30:
                raise OSError('less than 2 GiB host memory available')
            block, member_etag = exact_range(session, item['links']['self'],
                                              member['offset'], end, item['size'])
        data = decode_local_member(block, member)
        with h5py.File(io.BytesIO(data), 'r') as h:
            participants, trials, times = h['participant'][:], h['trial'][:], h['timestamp'][:]
            rates = h['vibration_sample_rate'][:]
        import numpy as np

        if (len(times) != len(participants) or len(rates) != len(participants)
                or not np.isfinite(times).all() or not np.isfinite(rates).all()
                or np.any(rates <= 0)):
            raise ValueError('invalid HDF5 metadata')
        summary = summarize_activity(participants, trials)
        result = {
            **protocol, 'status': 'PASS_MEMBER_CRC_SOURCE_FEASIBILITY_ONLY',
            'elapsed_seconds': time.perf_counter() - tick, 'tail_bytes': len(tail),
            'member_range_bytes': len(block), 'tail_etag': tail_etag, 'member_etag': member_etag,
            'member_name': member['name'], 'member_bytes': len(data),
            'member_sha256': hashlib.sha256(data).hexdigest(),
            'member_crc32': f"{member['crc']:08x}", 'rows': len(participants),
            'participant_count': len(set(map(int, participants))), 'split': summary,
            'rates_min': float(rates.min()), 'rates_max': float(rates.max()),
            'raw_member_retained': False, 'waveform_values_read': False,
            'other_four_sensors_checked': False, 'training_or_validation_scoring': False,
            'sealed_audit_loaded': False,
        }
        (args.output_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)
    except Exception:
        (args.output_dir / 'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
