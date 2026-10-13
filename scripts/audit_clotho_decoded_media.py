"""Complete-corpus decode audit and waveform cache; no model fitting."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from scripts.clotho_acquisition_gate import complete_acquisition


def check_waveform(waveform):
    if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("empty, malformed or nonfinite waveform")
    if not np.any(waveform):
        raise ValueError("all-zero waveform requires review")
    return hashlib.sha256(waveform.astype("<f4").tobytes()).hexdigest()


def main():
    import av

    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    media = root / "clotho-aqa-development-media-01"
    records = complete_acquisition(media, root / "clotho-aqa-development-split-01")
    output = root / "clotho-aqa-decoded-audit-01"
    output.mkdir(exist_ok=False)
    report = dict(
        status="RUNNING",
        rows=[],
        duplicate_pairs=[],
        sample_rate=2000,
        media_report_sha256=hashlib.sha256((media / "result.json").read_bytes()).hexdigest(),
        scope="exact decoded identity only; near-duplicate waveform audit still required",
    )
    seen = {}
    started = time.perf_counter()
    try:
        for row in records:
            with av.open(str(media / (row["source_id"] + ".wav"))) as container:
                converter = av.AudioResampler(format="fltp", layout="mono", rate=2000)
                arrays = []
                for frame in container.decode(audio=0):
                    arrays.extend(f.to_ndarray().ravel() for f in converter.resample(frame))
                arrays.extend(f.to_ndarray().ravel() for f in converter.resample(None))
            waveform = np.concatenate(arrays) if arrays else np.array([])
            digest = check_waveform(waveform)
            if digest in seen:
                report["duplicate_pairs"].append([seen[digest], row["source_id"]])
            seen[digest] = row["source_id"]
            np.save(output / (row["source_id"] + ".npy"), waveform)
            report["rows"].append(
                dict(
                    source_id=row["source_id"],
                    split=row["split"],
                    samples=len(waveform),
                    waveform_sha256=digest,
                )
            )
        if report["duplicate_pairs"]:
            raise ValueError("decoded duplicate media requires review before fitting")
        report["status"] = "PASS"
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
