"""Label-independent development groups and exhaustive raw TRAIN label accounting."""

import hashlib
from collections import Counter

from scripts.greatest_hits_source import ID


def partition_train(train, test):
    if (len(set(train)) != len(train) or len(set(test)) != len(test)
            or set(train) & set(test) or any(not ID.fullmatch(i) for i in train + test)):
        raise ValueError("unique, disjoint, safe official source IDs required")
    test_prefixes = {"-".join(i.split("-")[:6]) for i in test}
    quarantine = sorted(i for i in train if "-".join(i.split("-")[:6]) in test_prefixes)
    eligible = sorted(set(train) - set(quarantine))
    dates = sorted({i[:10] for i in eligible},
                   key=lambda s: hashlib.sha256(("17:" + s).encode()).hexdigest())
    if len(dates) <= 5:
        raise ValueError("more than five eligible dates required; do not change split policy")
    validation = [i for i in eligible if i[:10] in dates[:5]]
    return dict(train=sorted(set(eligible) - set(validation)), validation=validation,
                quarantine=quarantine)


def account_annotations(plan, annotations):
    ids = sum(plan.values(), [])
    if len(set(ids)) != len(ids) or set(ids) != set(annotations):
        raise ValueError("exact, disjoint parent accounting required")

    def summarize(parents):
        rows = [r for p in parents for r in annotations[p]]
        return dict(parents=len(parents), rows=len(rows),
                    complete=sum(None not in r["fields"] for r in rows),
                    missing_fields=dict(Counter(sum(x is None for x in r["fields"]) for r in rows)),
                    fields=[dict(sorted(Counter("<MISSING>" if r["fields"][i] is None
                                                else r["fields"][i] for r in rows).items()))
                            for i in range(3)])

    splits = {split: summarize(parents) for split, parents in plan.items()}
    all_rows = [r for rows in annotations.values() for r in rows]
    vocab = [sorted({r["fields"][i] for r in all_rows if r["fields"][i] is not None})
             for i in range(3)]
    absent = [sorted(set(splits["validation"]["fields"][i])
                     - set(splits["train"]["fields"][i]) - {"<MISSING>"}) for i in range(3)]
    parent_info = {p: dict(summarize([p]), distinct_known_labels=[len({r["fields"][i]
                    for r in annotations[p] if r["fields"][i] is not None}) for i in range(3)])
                   for p in sorted(annotations)}
    dates = {date: summarize([p for p in ids if p[:10] == date])
             for date in sorted({p[:10] for p in ids})}
    prefixes = {prefix: summarize([p for p in ids if "-".join(p.split("-")[:6]) == prefix])
                for prefix in sorted({"-".join(p.split("-")[:6]) for p in ids})}
    return dict(splits=splits, task_vocabularies=vocab, validation_values_absent_from_train=absent,
                parents=parent_info, dates=dates, timestamp_prefixes=prefixes)
