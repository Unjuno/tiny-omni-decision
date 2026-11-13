# Fixed text prefix factorial diagnostic

Fresh seed193,128 constructed states, six extremum questions/state, four arms:
familiar versus domain question body, crossed with absence/presence of the exact
prefix “Use cost as price, risk as danger, and time as timestamp.” All states,
targets and candidate order match across arms. Oracle verifies every target.
Zero group/content overlap with the explicitly hashed comparison-vocabulary train,
conditional validation and previous vocabulary-transfer inputs. This is a new
development diagnostic, not a blind audit or proof of independence from all prior
project data. No training, prompt selection or calibration.

Frozen coverage checkpoint b063f124e59c8a4c1b09c7bf2a8423846f73489d7ce8cd6649735745bc6cebf5;
unchanged independent-candidate scorer and tokenizer. Four fixed arms, all reported.

| Body / prefix | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Familiar / absent | .901042 | .248056 | .144571 | .041483 |
| Familiar / present | .513021 | 1.359645 | .702485 | .204126 |
| Domain / absent | .430990 | 1.168390 | .629336 | .106056 |
| Domain / present | .325521 | 1.463507 | .778221 | .158760 |

Prefix addition alone substantially changes fixed-model quality while preserving
the answer. Domain bodies are weak even without the prefix. Thus the old contrast
cannot be attributed solely to unfamiliar superlative vocabulary. Body changes
still combine vocabulary and syntax; no unique word-level explanation is claimed.
The prefix contains multiple fields, so field distraction is a hypothesis, not
yet separated from length/structure effects. These768 rows represent128 states;
arms do not multiply the independent state count. No product95% claim.

No runtime prompt stripping is adopted: removing potentially meaningful context
would alter the product contract. A future robustness intervention needs independent
training states and prospectively held-out distractor structures, including examples
where a preamble genuinely changes the requested decision. Preserve all four arms
as observed evidence; do not train on them or select the easiest prompt for release.

Artifacts: C:/CodexArtifacts/pretrained-reuse-r2/text-prefix-factorial-01.
Inputs, protocol, four prediction files and hashes are saved. Runner:
C:/CodexArtifacts/pretrained-reuse-r2/evaluate_prefix_factorial.py (hash in protocol).
Local CUDA inference only; GPU was checked before launch. No existing process was
stopped, checkpoint file changed, or paid compute used. Resource totals in JSON.
