"""One-step CPU embedding-delta integrity smoke; no quality claim."""


def main():
    import json
    from pathlib import Path

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from scripts.range_generalization import sha, write
    from scripts.train_conditional_repair import configure_repair_policy
    from scripts.train_text_crossencoder import frozen_hash
    from tiny_omni_decision.selected_embedding import SelectedEmbeddingDelta

    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    out = root / "numeric-embedding-smoke-01"
    assert not out.exists()
    path = Path("C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model")
    delta = root / "conditional-repair-comparison-vocabulary-01/best-delta.pt"
    assert sha(delta) == "868cc435f3b70ce84d0c49781d80680ba28a1078cbcf6de25f53a58756661b13"
    torch.set_num_threads(2)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).eval()
    names = configure_repair_policy(model, "all_encoder_blocks")
    saved = torch.load(delta, weights_only=True, map_location="cpu")
    assert set(names) == set(saved)
    model.load_state_dict(saved, strict=False)
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    r = json.loads(
        (root / "comparison-vocabulary-curriculum-01/train.jsonl").read_text().splitlines()[0]
    )
    inputs = tokenizer(
        [r["state"] + " " + r["question"]] * 4, r["options"], return_tensors="pt", padding=True
    )
    with torch.inference_mode():
        before = model(**inputs).logits.detach().clone()
    coverage = json.loads((root / "numeric-embedding-coverage-01/result.json").read_text())
    wrapper = SelectedEmbeddingDelta(
        model.get_input_embeddings(), [x["id"] for x in coverage["numeric_tokens"]]
    )
    model.set_input_embeddings(wrapper)
    basehash = frozen_hash(model)
    with torch.inference_mode():
        after = model(**inputs).logits
    assert torch.equal(before, after)
    optimizer = torch.optim.AdamW([wrapper.delta], lr=2e-5, weight_decay=0.01)
    logits = model(**inputs).logits[:, 0].reshape(1, 4)
    loss = torch.nn.functional.cross_entropy(
        logits, torch.tensor([r["options"].index(r["target"])])
    )
    loss.backward()
    assert torch.isfinite(wrapper.delta.grad).all() and wrapper.delta.grad.abs().sum() > 0
    assert all(
        p.grad is None
        for n, p in model.named_parameters()
        if n != "bert.embeddings.word_embeddings.delta"
    )
    norm = float(wrapper.delta.grad.norm())
    optimizer.step()
    assert frozen_hash(model) == basehash
    out.mkdir()
    result = dict(
        status="PASS_TECHNICAL_SMOKE_ONLY",
        examples=1,
        optimizer_updates=1,
        initial_logits_exact=True,
        base_unchanged=True,
        trainable_parameters=wrapper.delta.numel(),
        total_parameters=sum(p.numel() for p in model.parameters()),
        gradient_norm=norm,
        training_loss=float(loss.detach()),
        adopted=False,
        device="cpu",
        quality_evaluation=False,
        checkpoint_sha256=sha(delta),
        state_id=r["id"],
    )
    write(out / "result.json", result)
    write(Path("docs/pretrained_reuse/R2_NUMERIC_EMBEDDING_SMOKE_20261012.json"), result)
    print(result)


if __name__ == "__main__":
    main()
