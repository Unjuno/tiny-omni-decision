"""Bounded retry for atomic progress replacement under transient Windows read locks."""

import json
import time


def save_progress(path, report, *, attempts=11, delay=0.1):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for attempt in range(attempts):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
