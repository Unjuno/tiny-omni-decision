from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 2)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor([[0.0, 0.0], [1.0, 3.0], [3.0, 5.0], [99.0, 99.0]])
            )

    def forward(self, input_ids, attention_mask, return_dict):
        assert return_dict
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = Backbone()
        self.classifier = nn.Linear(2, 1)

    def forward(self, **kwargs):
        raise AssertionError("Classifier must not be used for feature extraction")


def test_shared_backbone_pooling_without_copy_or_update():
    from tiny_omni_decision.shared_text import shared_text_features

    model = Scorer().eval().requires_grad_(False)
    identities = [id(p) for p in model.parameters()]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    inputs = dict(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.tensor([[1, 1, 0]]))
    features = shared_text_features(model, inputs)
    assert torch.equal(features, torch.tensor([[2.0, 4.0]]))
    assert features.dtype == torch.float32 and not features.requires_grad
    assert identities == [id(p) for p in model.parameters()]
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert torch.equal(features, shared_text_features(model, inputs))


def test_shared_features_fail_closed_on_live_training_or_bad_mask():
    from tiny_omni_decision.shared_text import shared_text_features

    model = Scorer()
    inputs = dict(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.tensor([[1, 1]]))
    with pytest.raises(ValueError, match="frozen"):
        shared_text_features(model.eval(), inputs)
    model.requires_grad_(False).train()
    with pytest.raises(ValueError, match="evaluation"):
        shared_text_features(model, inputs)
    model.eval()
    for mask in [torch.tensor([[0, 0]]), torch.tensor([[1]]), torch.tensor([[1, -1]])]:
        with pytest.raises(ValueError):
            shared_text_features(model, dict(inputs, attention_mask=mask))
