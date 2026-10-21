"""Fixed context comparison; final checkpoint only, no validation selection."""

import argparse
import json
import os
import random
import time
from pathlib import Path

from scripts.range_generalization import sha, write


def ordered_views(rows):
    lookup = {(r["base_id"], r["context"]): r for r in rows}
    ids = sorted({r["base_id"] for r in rows})
    if len(ids) != 3072 or len(lookup) != 9216:
        raise ValueError("incomplete corpus")
    random.Random(17).shuffle(ids)
    kinds = ["plain", "irrelevant", "necessary"] * 1024
    random.Random(18).shuffle(kinds)
    return {
        "plain": [lookup[(i, "plain")] for i in ids],
        "context": [lookup[(i, k)] for i, k in zip(ids, kinds, strict=True)],
    }


def main():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from scripts.train_conditional_repair import configure_repair_policy
    from scripts.train_frozen_text_probe import _metrics
    from scripts.train_text_crossencoder import frozen_hash

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--policy", choices=["last_block", "all_encoder_blocks"], default="last_block"
    )
    parser.add_argument("--arm", choices=["both", "plain", "context"], default="both")
    args = parser.parse_args()
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    corpus = root / "context-robustness-corpus-01"
    manifest = json.loads((corpus / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha(corpus / name) != digest:
            raise ValueError("corpus changed")
    rows = [json.loads(s) for s in (corpus / "train.jsonl").read_text().splitlines()]
    dev = [json.loads(s) for s in (corpus / "development.jsonl").read_text().splitlines()]
    arms = ordered_views(rows)
    parent = root / "text-coverage-training-01/best-delta.pt"
    if sha(parent) != "b063f124e59c8a4c1b09c7bf2a8423846f73489d7ce8cd6649735745bc6cebf5":
        raise ValueError("parent changed")
    model_path = Path("C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model")
    if sha(model_path / "model.safetensors") != (
        "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
    ):
        raise ValueError("base changed")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False

    def run_arm(arm, train):
        out = root / f"context-robustness-{arm}-{'smoke' if args.smoke else 'run'}-01"
        if args.policy != "last_block":
            out = out.with_name(out.name + "-all-encoder")
        artifact_scope = "lastblock" if args.policy == "last_block" else "encoder"
        out.mkdir(exist_ok=False)
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        model = (
            AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            .cuda()
            .eval()
        )
        delta = torch.load(parent, weights_only=True, map_location="cpu")
        expected = {
            n
            for n, _ in model.named_parameters()
            if n.startswith(("bert.encoder.layer.", "classifier."))
        }
        if set(delta) != expected:
            raise ValueError("parent scope changed")
        model.load_state_dict(delta, strict=False)
        names = configure_repair_policy(model, args.policy)
        before = frozen_hash(model)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.01)
        steps = 2 if args.smoke else 384
        validation = dev[:24] if args.smoke else dev
        write(
            out / "config.json",
            dict(
                arm=arm,
                policy=args.policy,
                smoke=args.smoke,
                steps=steps,
                batch=8,
                seed=17,
                lr=2e-5,
                weight_decay=0.01,
                dropout=False,
                clip=1,
                selector="none; fixed final",
                parent_sha256=sha(parent),
                corpus_manifest_sha256=sha(corpus / "manifest.json"),
                runner_sha256=sha(Path(__file__)),
                trainable_names=names,
                trainable_parameters=sum(p.numel() for p in params),
                total_parameters=sum(p.numel() for p in model.parameters()),
                train_ids=[r["id"] for r in train[: steps * 8]],
                dev_ids=[r["id"] for r in validation],
                torch=torch.__version__,
            ),
        )

        def save(name):
            torch.save(
                {
                    n: p.detach().cpu().clone()
                    for n, p in model.named_parameters()
                    if p.requires_grad
                },
                out / name,
            )

        def forward(batch, reverse=False):
            first = [
                f"Question: {r['question']}\nCandidate answer: {o}"
                for r in batch
                for o in (r["options"][::-1] if reverse else r["options"])
            ]
            second = [f"State: {r['state']}" for r in batch for _ in r["options"]]
            tokens = tokenizer(first, second, padding=True, truncation=False, return_tensors="pt")
            if tokens["input_ids"].shape[1] > 512:
                raise ValueError("truncation required")
            logits = model(**tokens.to("cuda")).logits[:, 0].reshape(-1, 4)
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite logits")
            return logits

        history = []
        start = time.perf_counter()

        def evaluate(step):
            values = []
            with torch.inference_mode():
                for offset in range(0, len(validation), 8):
                    batch = validation[offset : offset + 8]
                    logits = forward(batch)
                    if step == steps:
                        torch.testing.assert_close(
                            logits, forward(batch, True).flip(-1), atol=1e-5, rtol=1e-5
                        )
                    values.extend(logits.cpu())
            metrics = {}
            for kind in ["plain", "irrelevant", "necessary"]:
                idx = [i for i, r in enumerate(validation) if r["context"] == kind]
                metrics[kind] = _metrics(
                    [values[i] for i in idx],
                    [validation[i]["options"].index(validation[i]["target"]) for i in idx],
                )
            write(
                out / f"predictions-{step}.json",
                [
                    dict(
                        id=r["id"],
                        group=r["group"],
                        context=r["context"],
                        options=r["options"],
                        target=r["target"],
                        probabilities=v.softmax(-1).tolist(),
                    )
                    for r, v in zip(validation, values, strict=True)
                ],
            )
            if frozen_hash(model) != before:
                raise ValueError("frozen tensors changed")
            history.append(dict(step=step, metrics=metrics, elapsed=time.perf_counter() - start))
            write(out / "history.json", history)
            print(arm, step, {k: v["accuracy"] for k, v in metrics.items()}, flush=True)
            return torch.stack(values)

        save(f"initial-{artifact_scope}.pt")
        evaluate(0)
        losses = []
        for step in range(1, steps + 1):
            batch = train[(step - 1) * 8 : step * 8]
            optimizer.zero_grad(set_to_none=True)
            logits = forward(batch)
            targets = torch.tensor([r["options"].index(r["target"]) for r in batch], device="cuda")
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                raise ValueError("invalid gradient")
            norm = torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
            optimizer.step()
            losses.append(dict(step=step, ce=loss.item(), gradient_norm=norm.item()))
            if step % 128 == 0 or step == steps:
                final = evaluate(step)
                write(out / "training.json", losses)
        save(f"final-{artifact_scope}.pt")
        model.load_state_dict(
            torch.load(out / f"final-{artifact_scope}.pt", weights_only=True), strict=False
        )
        with torch.inference_mode():
            replay = torch.cat(
                [forward(validation[i : i + 8]).cpu() for i in range(0, len(validation), 8)]
            )
        if not torch.equal(final, replay):
            raise ValueError("reload differs")
        write(
            out / "result.json",
            dict(
                status="PASS_FIXED_RUN",
                seconds=time.perf_counter() - start,
                reload_exact=True,
                frozen_unchanged=True,
                updates=steps,
                consumed=steps * 8,
                peak_vram=torch.cuda.max_memory_allocated(),
                artifacts={p.name: sha(p) for p in out.iterdir() if p.is_file()},
            ),
        )
        torch.cuda.empty_cache()

    for arm, train in arms.items():
        if args.arm in ("both", arm):
            run_arm(arm, train)


if __name__ == "__main__":
    main()
