"""Train-only MCQA modality-contract feasibility; lexical counts are not labels."""

import json
import zipfile
from collections import Counter
from pathlib import Path

from scripts.range_generalization import sha, write

p = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/perception-test-source-audit-01/train_annotations.zip"
)
assert sha(p) == "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78"
with zipfile.ZipFile(p) as archive:
    data = json.loads(archive.read("all_train.json"))
assert all(r["metadata"]["split"] == "train" for r in data.values())
questions = [q for r in data.values() for q in r["mc_question"]]
result = {
    "source_sha256": sha(p),
    "videos": len(data),
    "questions": len(questions),
    "fields": dict(Counter(k for q in questions for k in q)),
    "reasoning": dict(Counter(q["reasoning"] for q in questions)),
    "tags": dict(Counter(t for q in questions for t in q["tag"])),
    "lexical_sound_hear_audio_questions": sum(
        any(w in q["question"].lower() for w in ["sound", "hear", "audio"]) for q in questions
    ),
    "decision": (
        "No explicit required-modality supervision found; lexical absence is not proof "
        "of visual-only solvability. Do not auto-admit MCQA as complementary AV benchmark."
    ),
}
write(Path("docs/pretrained_reuse/R2_PERCEPTION_AV_QUESTION_AUDIT_20261013.json"), result)
