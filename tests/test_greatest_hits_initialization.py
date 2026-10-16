import copy

import pytest

torch = pytest.importorskip('torch')

def fixture(arm):
    from tiny_omni_decision.native_av_interaction import ConditionalAVReadout
    from tiny_omni_decision.native_candidate import NativeCandidateReadout

    torch.manual_seed(17)
    control = NativeCandidateReadout().state_dict()
    control['score.0.bias'] += .3  # Distinct simulated fitted baseline.
    torch.manual_seed(18)
    model = NativeCandidateReadout() if arm == 'C0' else ConditionalAVReadout(
        exchange=arm[1] == '1', gated=arm[3] == '1')
    return control, model.state_dict()


@pytest.mark.parametrize('arm', ['C0', 'E0G0', 'E0G1', 'E1G0', 'E1G1'])
def test_exact_control_copy_zero_residual_and_output_parity(arm):
    from scripts.greatest_hits_initialization import initialized_model
    from tiny_omni_decision.native_candidate import NativeTokenBlock

    control, state = fixture(arm)
    before = copy.deepcopy(control)
    model = initialized_model(arm, control, state).eval()
    baseline = model if arm == 'C0' else model.baseline
    for name, value in control.items():
        assert torch.equal(value, before[name])
        assert torch.equal(value, baseline.state_dict()[name])
    if arm != 'C0':
        for name, value in state.items():
            if not name.startswith('baseline.'):
                actual = model.state_dict()[name]
                assert (torch.equal(value, actual) if isinstance(value, torch.Tensor)
                        else value == actual)
    assert sum(p.numel() for p in model.parameters()) == (889345 if arm == 'C0' else 1732420)
    assert all(p.requires_grad for p in model.parameters())
    blocks = [NativeTokenBlock(k, torch.randn(5, d), torch.randn(5, 3),
                               torch.ones(5, dtype=torch.bool))
              for k, d in [('speech', 384), ('event_audio', 384), ('video', 768)]]
    queries = torch.randn(3, 384)
    with torch.no_grad():
        assert torch.equal(model(blocks, queries), baseline(blocks, queries))


@pytest.mark.parametrize('case', ['bad_arm', 'missing', 'nonfinite', 'wrong_dtype',
                                 'nonzero_output', 'wrong_policy'])
def test_invalid_initialization_fails_closed(case):
    from scripts.greatest_hits_initialization import initialized_model

    arm = 'E1G1'
    control, state = fixture(arm)
    if case == 'bad_arm':
        arm = 'E2G1'
    elif case == 'missing':
        del control['score.0.bias']
    elif case == 'nonfinite':
        control['score.0.bias'][0] = float('nan')
    elif case == 'wrong_dtype':
        control['score.0.bias'] = control['score.0.bias'].double()
    elif case == 'nonzero_output':
        state['correction.2.bias'] += 1
    else:
        state['_extra_state']['exchange'] = False
    with pytest.raises((ValueError, RuntimeError)):
        initialized_model(arm, control, state)
