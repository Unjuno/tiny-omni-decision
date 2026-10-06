# Postquant worker entrypoints

## Implemented boundary

Three thin scripts now share `src/tiny_omni_decision/postquant_launch.py`:

| Script | Default operation |
|---|---|
| `scripts/postquant_preflight.py` | Read-only launch-contract and artifact checks |
| `scripts/run_postquant_smoke.py` | Validate one trained-arm, 8-update smoke request |
| `scripts/run_postquant_comparison.py` | Validate exactly one Q0–Q4 arm |

**These scripts do not implement the numerical policy, quantizer, teacher cache,
postquant trainer, or exact resume.** At this revision the real trainer is absent.
A valid dry-run returns `VALIDATED_NOT_EXECUTED` and `backend_available: false`.
`--execute` fails closed rather than running legacy fixed-loss training or pretending
that a smoke experiment passed. There is no automatic all-arm execution.

Read the [execution plan](superpowers/plans/2026-10-06-post-quantization-respecialization-experiment.md)
and [review addendum](superpowers/plans/2026-10-06-postquant-review-addendum.md).
The existing fixed-loss YAML remains unchanged. Launch schema v1 has
`kind: postquant_launch`; it is **not** numerical-policy schema v2.

## Usage

Copy `configs/recovery/postquant_launch.example.json` outside the tracked source
checkout and replace every placeholder using the actual worker's pinned paths and
hashes. The checked-in template is deliberately not launchable and authorizes no
execution. Do not use an old checkpoint merely because its name says `best`.

```bash
python scripts/postquant_preflight.py --manifest /absolute/path/launch.json
python scripts/postquant_preflight.py --manifest /absolute/path/launch.json --check-device
python scripts/run_postquant_smoke.py --manifest /absolute/path/smoke-q3.json
python scripts/run_postquant_comparison.py --manifest /absolute/path/q3.json --arm q3_recovery_to_respecialization
```

Windows accepts native absolute paths such as `C:/Users/name/AppData/Local/CodexArtifacts`.
Quote paths containing spaces. `--repo` may select the source checkout to inspect.
The code's branch and full commit must match the manifest and the checkout must be
clean, including untracked files not excluded by Git. Keep launch manifests and
outputs outside tracked source or in explicitly ignored artifact locations.

All commands default to no execution. Preflight has no `--execute` option. Smoke
and comparison require **both** `--execute` and Boolean
`authorization.execute: true` for handoff. CLI option abbreviations, unknown flags,
arm/profile overrides and `--arm all` are rejected.

| Profile / arm | Total optimizer updates | Recovery updates |
|---|---:|---:|
| smoke Q2 | 8 | 0 (fixed objective) |
| smoke Q3/Q4 | 8 | 2 |
| comparison Q0/Q1 | 0 (evaluation only) | 0 |
| comparison Q2 | 1,024 | 0 (fixed objective) |
| comparison Q3/Q4 | 1,024 | 256 |

Gradient accumulation is 4 in this bounded contract. New research budgets require
a versioned contract, not an unnoticed command-line override. `--resume-from` is
currently rejected; no smoke checkpoint can become a longer run via these scripts.

## What preflight checks — and what it does not

It validates strict JSON (including duplicate keys, unknown fields and nonfinite
numbers), arm/profile budgets, local-only/no-download/no-paid-compute flags, Git
identity, explicit file/tree SHA-256 values, output nonexistence and free disk.
It rejects path traversal, URLs/UNC shares, symlink artifacts, protected
`audit`/`sealed`/`heldout` path components, and outputs inside frozen input trees.
It checks all supplied artifact paths before hashing their contents. There is no
recursive dataset search and no checkpoint deserialization.

`--check-device` queries `nvidia-smi` for the exact GPU name/index and total/free
VRAM; it does not import PyTorch or allocate CUDA tensors. Hardware is otherwise
reported as unverified. All actual handoffs require this hardware check and the
project's Python 3.11/3.12 range.

**Declared-file integrity is not a deep model/corpus audit.** The backend must
validate policy semantics, model/index/shard identity, sampler/option order,
media-reference safety, asset-level split integrity, runtime compatibility and
resource enforcement before opening data or allocating a model. A manifest
pointing to metadata does not imply its referenced weight shards were hashed.
Path checks prevent common mistakes; they are not a security sandbox against a
malicious local process or filesystem races. Use an immutable local input tree.

## Hashing contract

Files use SHA-256 over raw bytes. Trees use `postquant-tree-v1`: SHA-256 initialized
with UTF-8 `postquant-tree-v1\n`, then one compact ASCII-escaped JSON line per file,
sorted by relative POSIX path. Each line is `[relative_path,file_sha256]` plus a
newline. Empty trees and links are rejected; directories are not independent
entries. Do not substitute another tree-hash convention silently.

The helper `artifact_hash(Path(...))` computes this digest, without loading weights.
Hash only explicitly selected, non-audit training artifacts. Never copy credentials
or machine-local manifests into Git.

## Backend integration gate

The fixed module `tiny_omni_decision.postquant_trainer` must be implemented at the
inspected checkout path. An arbitrary module or shell command cannot be supplied.
Before handoff it must expose:

```python
POSTQUANT_LAUNCH_API_VERSION = 1
POSTQUANT_LAUNCH_CAPABILITIES = {
    "local_only", "offline", "resource_caps", "exclusive_output", "launch_v1"
}
# Existing execution-plan signature:
def run_postquant_experiment(*, arm, run_manifest, output_dir, resume_from=None):
    ...
```

These markers are a compatibility declaration, not proof of correctness. Add them
only with real integration tests for deep validation, resource caps, exclusive
output creation and numerical loss/schedule wiring. Module import must be
side-effect-free. The launcher sets offline library flags before import and
restores them on success/failure; these flags are not an OS network firewall.

The backend owns output creation and failure records. A sibling `.launch.lock`
prevents cooperating launchers from racing; existing runs/locks are never
silently removed. A crash can leave a lock: inspect the process and run state
before manual recovery. Backend failures propagate and their output is retained.
`BACKEND_RETURNED` is not a claim of quality, 95% qualification or a passed audit.

## Verification and evidence

The tests use temporary Git repositories and tiny synthetic files. The simulated
handoff test uses a test double, not a model. No GPU run, ternary conversion,
paid compute, model download or final-audit access is part of this test suite.

```bash
python -m pytest tests/test_postquant_launch.py -q
```

Implementation references (checked while writing):
- Python 3.11 argparse: https://docs.python.org/3.11/library/argparse.html
- Python 3.11 pathlib: https://docs.python.org/3.11/library/pathlib.html
- Python 3.11 subprocess: https://docs.python.org/3.11/library/subprocess.html
- PyTorch checkpoint guidance: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html

The last reference is a downstream requirement: restoring weights alone is not
exact training resume. This change intentionally does not deserialize a checkpoint.
