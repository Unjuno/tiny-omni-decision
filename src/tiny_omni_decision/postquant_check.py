"""Offline policy inspection and optional tiny CPU numerical checks.

Never imports transformers/PEFT, loads a checkpoint/corpus, or steps an optimizer.
This is not the 8-update real-model GPU smoke and cannot enable its launch gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .postquant_policy import load_postquant_config, loss_weights_for_update, policy_identity

def synthetic_cpu_checks() -> dict[str, Any]:
    import torch
    from .postquant_loss import option_distillation_loss
    from .postquant_policy import LossWeights
    from .ternary_reference import TernaryTensor, dequantize_reference, quantize_reference
    student = torch.zeros(1, 2, device='cpu', requires_grad=True)
    teacher = torch.zeros(1, 2, device='cpu', requires_grad=True)
    weights = LossWeights(stage='recovery', option_kl=1.0, cross_entropy=0.2, brier=0.2)
    loss = option_distillation_loss(student,
         teacher,
         torch.tensor([1],
         device='cpu'),
         torch.ones(1,
         2,
         dtype=torch.bool,
         device='cpu'),
         weights)
    expected = {'option_kl': 0.0, 'cross_entropy': 0.6931471805599453, 'brier': 0.5}
    for key, value in expected.items():
        if abs(loss[key].item() - value) > 1e-06:
            raise ValueError(f'synthetic CPU loss check failed: {key}')
    loss['total'].backward()
    if teacher.grad is not None or not bool(torch.isfinite(student.grad).all()):
        raise ValueError('synthetic CPU gradient check failed')
    source = torch.tensor([[-1.0, 0.0, 1.0, 0.25]], device='cpu')
    reference = quantize_reference(source, group_size=4)
    restored = TernaryTensor.from_state_dict(reference.to_state_dict())
    actual = dequantize_reference(restored, dtype=torch.float32)
    if not torch.equal(actual, torch.tensor([[-1.0, 0.0, 1.0, 0.0]], device='cpu')):
        raise ValueError('synthetic CPU ternary roundtrip failed')
    return {'torch': torch.__version__,
         'device': 'cpu',
         'data': 'synthetic_constants_only',
         'uniform_two_choice': {key: float(value.detach()) for key,
         value in loss.items()},
         'teacher_gradient_absent': True,
         'quantizer_roundtrip': True,
         'reference_storage': reference.storage_report(),
         'optimizer_updates': 0}

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--completed-updates', type=int, default=0)
    parser.add_argument('--check-numerics',
         action='store_true',
         help='Use tiny synthetic CPU tensors; no model, optimizer or GPU')
    args = parser.parse_args(argv)
    try:
        config = load_postquant_config(args.policy)
        weights = loss_weights_for_update(args.completed_updates, config)
        report = {'status': 'POLICY_VALIDATED_NOT_EXECUTED',
             'training_started': False,
             'model_loaded': False,
             'gpu_used': False,
             'arm': config.arm,
             'profile': config.profile,
             'policy_sha256': policy_identity(config),
             'completed_updates': args.completed_updates,
             'weights': weights.model_dump(),
             'last_update_weights': loss_weights_for_update(config.total_updates - 1,
             config).model_dump(),
             'checks': synthetic_cpu_checks() if args.check_numerics else None,
             'limitation': 'Gemma trainer/cache/model conversion/exact resume remain unverified'}
        print(json.dumps(report, allow_nan=False, indent=2))
        return 0
    except (ValueError, TypeError, OSError, ImportError, RuntimeError) as exc:
        print(json.dumps({'status': 'BLOCKED', 'error': str(exc)}), file=sys.stderr)
        return 2
