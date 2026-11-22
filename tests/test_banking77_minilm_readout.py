import torch

from scripts.train_banking77_minilm_readout import metric_summary, train_readout


def test_metric_summary_reports_finite_multiclass_metrics():
    labels = torch.tensor([0, 1, 2, 0])
    logits = torch.tensor(
        [
            [4.0, 0.0, -1.0],
            [0.0, 3.0, -1.0],
            [-1.0, 0.0, 2.0],
            [0.0, 4.0, -1.0],
        ]
    )
    metrics = metric_summary(logits, labels)

    assert metrics["accuracy"] == 0.75
    assert 0.0 < metrics["nll"]
    assert 0.0 < metrics["brier"]
    assert 0.0 <= metrics["ece_15"] <= 1.0
    assert 0.0 <= metrics["macro_recall"] <= 1.0


def test_early_stopping_keeps_best_validation_checkpoint():
    generator = torch.Generator().manual_seed(21)
    features = torch.randn((180, 8), generator=generator)
    labels = torch.arange(180) % 3
    features[:, 0] += labels.float() * 2.0
    train_readout_model, fit = train_readout(
        features[:135],
        labels[:135],
        features[135:],
        labels[135:],
        seed=17,
        learning_rate=0.02,
        batch_size=45,
        max_epochs=30,
        patience=4,
    )

    assert fit["best_epoch"] <= fit["epochs_completed"] <= 30
    assert fit["best_validation_metrics"]["nll"] == min(
        epoch["validation"]["nll"] for epoch in fit["history"]
    )
    assert train_readout_model.training is False
