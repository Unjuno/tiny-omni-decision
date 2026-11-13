"""Label-free index for the six frozen development text files only."""

import hashlib
import json

from scripts.native_training_readiness import TEXT, state_key


def validate_profile(config):
    expected = dict(window_content_tokens=510, overlap_content_tokens=32, batch_size=1,
                    dtype='float32', attention='eager', seed=17)
    if any(config.get(k) != value for k, value in expected.items()):
        raise ValueError('configuration differs from the implemented pinned profile')


def text_key(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def pair_key(question, candidate):
    return text_key(json.dumps([question, candidate], ensure_ascii=False, separators=(",", ":")))


def inventory(files):
    if set(files) != {name for names in TEXT.values() for name in names}:
        raise ValueError("exactly the six registered train/validation files required")
    states, pairs, mapped = {}, {}, {}
    groups = {s: set() for s in TEXT}
    normalized = {s: set() for s in TEXT}
    seen = set()
    for split, names in TEXT.items():
        for name in names:
            mapped[name] = []
            if not files[name]:
                raise ValueError("empty source file")
            for row in files[name]:
                if (row["id"] in seen or row["split"] != split
                        or not isinstance(row["state"], str) or not row["state"].strip()
                        or not isinstance(row["question"], str) or not row["question"].strip()
                        or not 2 <= len(row["options"]) <= 40
                        or any(not isinstance(x, str) or not x.strip() for x in row["options"])):
                    raise ValueError("invalid, duplicate or wrong-split source row")
                seen.add(row["id"])
                groups[split].add(row["group"])
                normalized[split].add(state_key(row["state"]))
                key = text_key(row["state"])
                states[key] = row["state"]
                queries = []
                for option in row["options"]:
                    query = pair_key(row["question"], option)
                    pairs[query] = (row["question"], option)
                    queries.append(query)
                mapped[name].append(dict(id=row["id"], state=key, queries=queries))
    if groups["train"] & groups["validation"] or normalized["train"] & normalized["validation"]:
        raise ValueError("cross-split group or normalized-state overlap")
    return dict(states=dict(sorted(states.items())), pairs=dict(sorted(pairs.items())), rows=mapped)


def storage_plan(index, tokenizer):
    states, queries, shapes = {}, {}, {}
    for split, names in TEXT.items():
        keys = sorted({r['state'] for name in names for r in index['rows'][name]})
        offset = 0
        filename = split + '-states.npy'
        for key in keys:
            size = len(tokenizer(index['states'][key], add_special_tokens=True,
                                 truncation=False)['input_ids'])
            if not 2 <= size <= 8192:
                raise ValueError('state exceeds full readout token budget')
            states[key] = dict(file=filename, offset=offset, tokens=size, split=split)
            offset += size
        shapes[filename] = [offset, 384]
    for offset, (key, (question, candidate)) in enumerate(index['pairs'].items()):
        size = len(tokenizer(question, text_pair=candidate, add_special_tokens=True,
                             truncation=False)['input_ids'])
        if not 2 <= size <= 512:
            raise ValueError('question/candidate exceeds encoder context')
        queries[key] = dict(file='queries.npy', offset=offset, tokens=size)
    shapes['queries.npy'] = [len(queries), 384]
    return dict(states=states, queries=queries, rows=index['rows'], shapes=shapes,
                tensor_bytes=sum(n * d * 4 for n, d in shapes.values()))
