"""Count actual live parameter objects, never deduplicate equal file contents."""


def parameter_inventory(models):
    seen = set()
    rows = {}
    for name, model in models.items():
        parameters = list(model.parameters())
        fresh = [p for p in parameters if id(p) not in seen]
        seen.update(id(p) for p in parameters)
        rows[name] = dict(logical_parameters=sum(p.numel() for p in parameters),
                          incremental_parameters=sum(p.numel() for p in fresh),
                          incremental_bytes=sum(p.numel() * p.element_size() for p in fresh))
    return dict(components=rows,
                total_parameters=sum(r["incremental_parameters"] for r in rows.values()),
                tensor_bytes=sum(r["incremental_bytes"] for r in rows.values()))
