"""One predeclared seed17 CLAP representation comparison; no encoder training."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from scripts.clap_projection import fit_train_projection, project, validate_coverage
from scripts.clotho_acquisition_gate import complete_acquisition
from scripts.extract_clap_full_clip import CONFIG_SHA, PROCESSOR_SHA, WEIGHT_SHA
from scripts.train_clotho_audio_comparison import fit, load_inputs

ROOT = Path('C:/CodexArtifacts/pretrained-reuse-r2')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output, features):
    raw, y, rows, baseline = load_inputs(ROOT)
    records = complete_acquisition(
        ROOT / 'clotho-aqa-development-media-02', ROOT / 'clotho-aqa-development-split-01'
    )
    info = json.loads((features / 'result.json').read_text(encoding='utf-8'))
    for key, expected in (
        ('weight_sha256', WEIGHT_SHA), ('config_sha256', CONFIG_SHA),
        ('processor_sha256', PROCESSOR_SHA),
        ('media_report_sha256', baseline['audio']['media_report_sha256']),
    ):
        if info[key] != expected:
            raise ValueError('CLAP encoder/processor/media pin differs')
    if (info['parameters'] != 28190872 or info['sample_rate'] != 48000
            or info['window_samples'] != 480000 or info['training'] is not False
            or info['dtype'] != 'float32'):
        raise ValueError('encoder or full-clip representation changed')
    for name, key in (('features.npy', 'feature_sha256'),
                      ('window_features.npy', 'window_feature_sha256')):
        if digest(features / name) != info[key]:
            raise ValueError('full-clip feature hash differs')
    audio = np.load(features / 'features.npy', allow_pickle=False)
    windows = np.load(features / 'window_features.npy', allow_pickle=False)
    if audio.shape != (856, 512):
        raise ValueError('expected complete 856-source CLAP representation')
    validate_coverage(records, info, audio, windows)
    source_ids = [r['source_id'] for r in records]
    train_ids = [r['source_id'] for r in records if r['split'] == 'train']
    if len(train_ids) != 578:
        raise ValueError('fixed unique training source count changed')
    projection = fit_train_projection(audio, source_ids, train_ids, 384)
    reduced = project(audio, projection)
    output.mkdir(exist_ok=False)
    np.savez(output / 'projection.npz', mean=projection['mean'],
             components=projection['components'], singular_values=projection['singular_values'])
    np.save(output / 'projected_features.npy', reduced)
    lookup = {identity: i for i, identity in enumerate(source_ids)}
    raw = {split: (reduced[[lookup[r['source_id']] for r in rows[split]]], pair[1])
           for split, pair in raw.items()}
    provenance = dict(
        baseline=baseline, audio=info, feature_report_sha256=digest(features / 'result.json'),
        projection_fit_ids=projection['fit_ids'], projection_dimensions=384,
        projection_retained_variance=projection['retained_variance'],
        projection_sha256=digest(output / 'projection.npz'),
        projected_feature_sha256=digest(output / 'projected_features.npy'),
        projection_resident_coefficients=int(projection['mean'].size
                                             + projection['components'].size),
        audio_encoder_parameters=28190872, question_encoder_parameters=17389824,
        readout_parameters=49409, combined_parameters_including_projection=45827225,
        scope='two-input audio/question development probe, not complete Omni runtime',
        pretraining_caveat='upstream checkpoint selection used Clotho retrieval; '
                          'HF checkpoint ancestry/record overlap UNKNOWN; not blind',
        selection_caveat='same reused validation selects checkpoint; no blind audit',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        runner_sha256=digest(Path(__file__)),
        projection_helper_sha256=digest(Path(__file__).with_name('clap_projection.py')),
        learning_code_sha256=digest(Path(__file__).with_name('train_clotho_audio_comparison.py')),
        options=['no', 'yes'], seed=17, epochs=24, batch_size=128, hidden_size=64,
        learning_rate=.001, weight_decay=.001,
        selector='earliest minimum raw validation NLL',
        calibration='17 geometric temperatures .25..4, calibration split only',
        frozen_encoders=True, no_new_seeds_or_budget_search=True,
    )
    (output / 'protocol.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return raw, y, rows, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    raw, y, rows, provenance = prepare(args.output, args.features)
    if not args.prepare_only:
        for condition in ('audio', 'joint'):
            fit(ROOT, condition, raw, y, rows, provenance,
                output_directory=args.output / condition)
    print(json.dumps(dict(status='PREPARED' if args.prepare_only else 'COMPLETE',
                          retained_variance=provenance['projection_retained_variance'],
                          parameters=provenance['combined_parameters_including_projection'])))


if __name__ == '__main__':
    main()
