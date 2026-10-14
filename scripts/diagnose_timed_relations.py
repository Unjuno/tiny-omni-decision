"""Input-only all-candidate spatial routing, then separately admitted posthoc scoring."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

from scripts.probe_timed_mask_tracking import (
    reconstruct_queries,
    require_arrays,
    require_bundle_binding,
    verify_sources,
)
from scripts.probe_track_appearance import ROOT, sha, write
from tiny_omni_decision.proposal_opportunity import source_box_at_time
from tiny_omni_decision.spatial_relations import (
    future_relation_summary,
    nearest_anchors,
    seed_box_roles,
)
from tiny_omni_decision.track_association import group_membership

TRACK_PIN = '1ed0d7e4b3114be53fd6df7359d895f04f2e361579d052231f7c95b4e96d1c05'
PLAN_PIN = 'a683e5a6f9772ca1cde37eec10419364707b9f2f295bb282ca62cfab4ce2fee0'
ANNOTATION_PIN = '9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78'
ARMS = ('actual', 'static', 'reversed')


def admit_inputs():
    run = ROOT / 'dynamic-timed-mask-tracks-01'
    path = run / 'result.json'
    if sha(path) != TRACK_PIN:
        raise ValueError('pinned completed full timed-tracking run required')
    result = json.loads(path.read_text())
    cfg_path = run / 'effective-config.json'
    if (result['status'] != 'PASS_ALL_TIMED_TRACKS_NOT_QA'
            or sha(cfg_path) != result['config_sha256'] or sha(result['path']) != result['sha256']):
        raise ValueError('protected full tracking/config/array bundle changed')
    cfg = json.loads(cfg_path.read_text())
    verify_sources(cfg['source_hashes'])
    plan_path = ROOT / 'dynamic-timed-mask-plan-02/result.json'
    if sha(plan_path) != PLAN_PIN or cfg['query_plan_sha256'] != PLAN_PIN:
        raise ValueError('fixed corrected query plan required')
    plan = json.loads(plan_path.read_text())
    contract, expected = reconstruct_queries()
    with np.load(plan['queries_path'], allow_pickle=False) as saved:
        require_bundle_binding(plan, saved, contract, expected)
    with np.load(result['path'], allow_pickle=False) as saved:
        for key in ('query_tyx', 'proposal_indices', 'frame_indices', 'on_or_after_observation'):
            require_arrays({key: saved[key]}, {key: expected[key]})
        tracks = saved['tracks'].copy()
    if (tracks.shape != (1070, 558, 2) or tracks.dtype != np.float32
            or not np.isfinite(tracks).all()):
        raise ValueError('complete finite raw track history required')
    return result, cfg, plan, expected, tracks


def input_predictions(tracks, queries, times):
    anchors = np.flatnonzero(queries[:, 0] == 0).astype(np.int64)
    if len(anchors) != 40 or not np.array_equal(anchors, np.arange(40)):
        raise ValueError('all40 first-frame anchors required; no label/score filter')
    xy = tracks[:, times] / np.array([256, 256], np.float32)
    a = xy[anchors]
    variants = dict(actual=a, static=np.broadcast_to(a[:, :1], a.shape), reversed=a[:, ::-1])
    predictions = {arm: nearest_anchors(xy, value) for arm, value in variants.items()}
    return xy, anchors, predictions


def predict(output, reference):
    start = time.perf_counter()
    if shutil.disk_usage(output).free < 8 * 1024**3:
        raise OSError('fixed8GiB disk reserve required')
    result, cfg, plan, expected, tracks = admit_inputs()
    xy, anchors, predictions = input_predictions(tracks, expected['query_tyx'],
                                                expected['frame_indices'])
    arrays = {k: v for k, v in expected.items() if k != 'nonempty'}
    arrays.update(query_xy=xy, anchor_query_indices=anchors)
    for arm, values in predictions.items():
        arrays.update({arm + '_' + key: value for key, value in values.items()})
    dest = output / 'predictions.npz'
    np.savez_compressed(dest, **arrays)
    with np.load(dest, allow_pickle=False) as saved:
        require_arrays(saved, arrays)
    if reference:
        previous = json.loads((reference / 'result.json').read_text())
        if (previous['status'] != 'PASS_ALL_NEAREST_SPATIAL_PREDICTIONS_NOT_QA'
                or sha(previous['prediction_path']) != previous['prediction_sha256']):
            raise ValueError('completed unchanged primary prediction bundle required')
        with np.load(previous['prediction_path'], allow_pickle=False) as saved:
            require_arrays(saved, arrays)
    verify_sources(cfg['source_hashes'])
    if sha(ROOT / 'dynamic-timed-mask-tracks-01/result.json') != TRACK_PIN:
        raise ValueError('protected tracking result changed')
    report = dict(status='PASS_ALL_NEAREST_SPATIAL_PREDICTIONS_NOT_QA',
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                        text=True).strip(),
                  runner_sha256=sha(Path(__file__)),
                  helper_sha256=sha(Path('src/tiny_omni_decision/spatial_relations.py')),
                  track_result_sha256=TRACK_PIN, query_plan_sha256=PLAN_PIN,
                  prediction_path=str(dest), prediction_sha256=sha(dest),
                  queries=1070, anchors=40, times=32, all_prediction_cases_per_arm=34240,
                  arrays_readback_bitwise=True, fresh_replay_bitwise=bool(reference),
                  exact_distance_tie_counts={a: int((v['ties'] > 1).sum())
                                             for a, v in predictions.items()},
                  annotations_loaded=False, training=False, validation_opened=False,
                  sealed_audit_opened=False, gpu_or_model_loaded=False, cloud_cost=0,
                  python=platform.python_version(), numpy=np.__version__,
                  pid=os.getpid(), seconds=time.perf_counter() - start)
    write(output / 'result.json', report)
    print(report['status'], report['seconds'], flush=True)


def admit_predictions(prediction_run):
    path = prediction_run / 'result.json'
    result_pin = sha(path)
    report = json.loads(path.read_text())
    if (report['status'] != 'PASS_ALL_NEAREST_SPATIAL_PREDICTIONS_NOT_QA'
            or report['runner_sha256'] != sha(Path(__file__))
            or report['helper_sha256'] != sha(Path('src/tiny_omni_decision/spatial_relations.py'))
            or sha(report['prediction_path']) != report['prediction_sha256']):
        raise ValueError('complete unchanged prediction bundle mandatory before labels')
    _, _, plan, expected, tracks = admit_inputs()
    xy, anchors, predictions = input_predictions(tracks, expected['query_tyx'],
                                                expected['frame_indices'])
    arrays = {k: v for k, v in expected.items() if k != 'nonempty'}
    arrays.update(query_xy=xy, anchor_query_indices=anchors)
    for arm, values in predictions.items():
        arrays.update({arm + '_' + key: value for key, value in values.items()})
    with np.load(report['prediction_path'], allow_pickle=False) as saved:
        require_arrays(saved, arrays)
    return report, result_pin, plan, arrays


def grounded_roles(source):
    roles = {}
    for role, suffix in dict(hidden='Track the occluded object.',
                             cover='Track the object that occludes the smaller object.',
                             group='Track these similar or identical objects.').items():
        matches = [q for q in source['grounded_question'] if q['question'].endswith(suffix)]
        if len(matches) != 1:
            raise ValueError('exact source-defined role required: ' + role)
        roles[role] = matches[0]['answers']
        if (not roles[role] or len(set(roles[role])) != len(roles[role])
                or any(not isinstance(i, int) or isinstance(i, bool) or i < 0
                       for i in roles[role])):
            raise ValueError('unique integer grounded IDs required')
    if len(roles['hidden']) != 1 or len(roles['cover']) != 1:
        raise ValueError('one source hidden/cover role required')
    if roles['cover'][0] not in roles['group'] or roles['hidden'][0] in roles['group']:
        raise ValueError('distinct grounded hidden and covering/group namespaces required')
    return roles


def score(output, prediction_run):
    start = time.perf_counter()
    report, prediction_pin, plan, arrays = admit_predictions(prediction_run)
    archive_path = ROOT / 'perception-test-source-audit-01/train_annotations.zip'
    if sha(archive_path) != ANNOTATION_PIN:
        raise ValueError('pinned training-source archive required')
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ['all_train.json']:
            raise ValueError('unexpected annotation archive')
        source = json.loads(archive.read('all_train.json'))['video_10903']
    if (source['metadata']['split'] != 'train'
            or source['metadata']['video_id'] != 'video_10903'):
        raise ValueError('fixed train source only')
    roles = grounded_roles(source)
    identities = roles['group'] + roles['hidden']
    source_tracks = {}
    for identity in identities:
        matches = [t for t in source['object_tracking'] if t['id'] == identity]
        if len(matches) != 1:
            raise ValueError('exact grounded source track required')
        source_tracks[identity] = matches[0]
    query, times = arrays['query_tyx'], arrays['frame_indices']
    seed_labels = np.full(1070, -1, np.int64)
    source_cases = []
    for row in plan['rows']:
        frame, first, count = row['frame_index'], row['offset'], row['count']
        frame_path = ROOT / f'dynamic-discovery-fixed32-01/{frame:04d}-proposals.npz'
        if sha(frame_path) != row['bank_sha256']:
            raise ValueError('protected proposal source changed')
        with np.load(frame_path, allow_pickle=False) as saved:
            boxes = saved['boxes'][arrays['proposal_indices'][first:first + count]]
        targets = {i: source_box_at_time(source_tracks[i], frame) for i in identities}
        available = [i for i in identities if targets[i]['box'] is not None]
        seed_labels[first:first + count] = seed_box_roles(
            boxes, query[first:first + count, [2, 1]] / 256,
            np.asarray(available, np.int64),
            np.asarray([targets[i]['box'] for i in available], np.float64).reshape(-1, 4))
        source_cases.append(dict(frame_index=frame, queries=count,
                                 source_proxies={str(i): t for i, t in targets.items()}))
    anchors = arrays['anchor_query_indices']
    anchor_initial = seed_labels[anchors]
    anchor_current, group_available = group_membership(
        source['object_tracking'], roles['group'], arrays['query_xy'][anchors], times)
    hidden_current, hidden_available = group_membership(
        source['object_tracking'], roles['hidden'], arrays['query_xy'], times)
    hidden_queries = np.flatnonzero(seed_labels == roles['hidden'][0])
    group_queries = np.flatnonzero(np.isin(seed_labels, roles['group']))
    later_group_queries = group_queries[query[group_queries, 0] > 0]
    initial_group_queries = group_queries[query[group_queries, 0] == 0]
    metrics = {}
    labels = dict(seed_labels=seed_labels, anchor_initial_source_role=anchor_initial,
                  anchor_current_source_role=anchor_current,
                  group_source_available=group_available, hidden_source_available=hidden_available,
                  hidden_query_indices=hidden_queries, group_query_indices=group_queries,
                  later_group_query_indices=later_group_queries,
                  initial_group_query_indices=initial_group_queries,
                  hidden_current_source_role=hidden_current)
    for arm in ARMS:
        selected = arrays[arm + '_indices']
        initial = anchor_initial[selected]
        current = anchor_current[selected, np.arange(len(times))[None]]
        labels[arm + '_initial_source_role'] = initial
        labels[arm + '_current_source_role'] = current
        metrics[arm] = {}
        roles_to_score = (('hidden_to_cover', hidden_queries), ('group_to_same', group_queries),
                          ('later_group_to_same', later_group_queries),
                          ('initial_anchor_to_same', initial_group_queries))
        for role, selected_q in roles_to_score:
            wanted = (np.full(len(selected_q), roles['cover'][0], np.int64)
                      if role == 'hidden_to_cover' else seed_labels[selected_q])
            metrics[arm][role] = {
                proxy: future_relation_summary(wanted, values[selected_q],
                                               query[selected_q, 0].astype(np.int64), times)
                for proxy, values in dict(initial_anchor_role=initial,
                                          current_anchor_bbox_role=current).items()}
    retention = future_relation_summary(
        np.full(len(hidden_queries), roles['hidden'][0], np.int64),
        hidden_current[hidden_queries], query[hidden_queries, 0].astype(np.int64), times)
    dest = output / 'posthoc-labels.npz'
    np.savez_compressed(dest, **labels)
    with np.load(dest, allow_pickle=False) as saved:
        require_arrays(saved, labels)
    if (sha(prediction_run / 'result.json') != prediction_pin
            or sha(report['prediction_path']) != report['prediction_sha256']
            or sha(archive_path) != ANNOTATION_PIN):
        raise ValueError('protected predictions/annotations changed during scoring')
    result = dict(status='PASS_POSTHOC_SPATIAL_RELATION_DIAGNOSTIC_NOT_QA',
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                        text=True).strip(),
                  runner_sha256=sha(Path(__file__)), prediction_result_sha256=prediction_pin,
                  annotation_sha256=ANNOTATION_PIN, source_roles=roles, source_cases=source_cases,
                  metrics=metrics, hidden_track_bbox_retention=retention,
                  all_queries=1070, seed_known_group_queries=len(group_queries),
                  seed_known_hidden_queries=len(hidden_queries),
                  seed_unknown_queries=int((seed_labels == -1).sum()),
                  seed_ambiguous_queries=int((seed_labels == -2).sum()),
                  hidden_queries=[dict(query_index=int(i), frame=int(query[i, 0]),
                                       proposal_id=int(arrays['proposal_indices'][i]))
                                  for i in hidden_queries],
                  labels_path=str(dest), labels_sha256=sha(dest),
                  annotations_opened_after_complete_prediction_admission=True,
                  training=False, validation_opened=False, sealed_audit_opened=False,
                  gpu_or_model_loaded=False, cloud_cost=0, seconds=time.perf_counter() - start)
    write(output / 'result.json', result)
    print(result['status'], result['seconds'], json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--score', type=Path, help='completed input-only prediction run')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    start = time.perf_counter()
    try:
        if args.score:
            if args.reference:
                raise ValueError('reference replay is prediction-only')
            score(args.output, args.score)
        else:
            predict(args.output, args.reference)
    except Exception:
        write(args.output / 'failure.json', dict(status='FAIL_INCOMPLETE',
              mode='posthoc_score' if args.score else 'input_only_predict',
              seconds=time.perf_counter() - start, error=traceback.format_exc(),
              runner_sha256=sha(Path(__file__))))
        raise


if __name__ == '__main__':
    main()
