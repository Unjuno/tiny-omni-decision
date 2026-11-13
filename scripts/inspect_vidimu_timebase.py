"""Compare processed time bases for the already inspected training trial."""

import hashlib
import zipfile
from pathlib import Path

from scripts.range_generalization import sha, write


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    path = root / "vidimu-source-audit-02/dataset.zip"
    if sha(path) != "605b695dcec5c2e7c4fd9a81dda22b60049697063d9eda150a637166813b79c2":
        raise ValueError("archive changed")
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in ["dataset/videoandimus/S40/ik_S40_A01_T01.mot",
                     "dataset/videoandimus/S40/S40_A01_T01.sto"]:
            data = archive.read(name)
            lines = [line for line in data.decode().splitlines() if line.strip()]
            header = lines.index("endheader")
            times = [float(line.split()[0]) for line in lines[header + 2:]]
            deltas = [b - a for a, b in zip(times, times[1:], strict=False)]
            result[name] = {"sha256": hashlib.sha256(data).hexdigest(),
                            "rows": len(times), "first": times[0], "last": times[-1],
                            "minimum_delta": min(deltas), "maximum_delta": max(deltas)}
    output = root / "vidimu-training-timebase-audit-01"
    output.mkdir(exist_ok=False)
    write(output / "result.json", result)
    print(result)


if __name__ == "__main__":
    main()
