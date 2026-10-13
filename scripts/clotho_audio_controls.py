"""Label-independent source-coherent audio derangement for inference diagnostics."""

from scripts.clotho_aqa_split import normalized
from scripts.starss23_readout_common import different_recording_permutation

VALIDATION_MAPPING_SHA256 = "82b6a398c80dbdc8fc0519262e82863454b033afb0b624789d9b4dbda38e4efd"


def source_derangement(rows, seed=17):
    owners = {}
    for row in rows:
        identity, owner = row["source_id"], normalized(row["uploader"])
        if not identity.isdecimal() or not owner:
            raise ValueError("source/uploader identity required")
        if identity in owners and owners[identity] != owner:
            raise ValueError("source has conflicting uploader identities")
        owners[identity] = owner
    sources = sorted(owners)
    permutation = different_recording_permutation([owners[s] for s in sources], seed)
    return {
        source: sources[replacement]
        for source, replacement in zip(sources, permutation, strict=True)
    }
