"""Cross-process Path A dispatcher for the independent local runtime profiles."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from tiny_omni_decision.modular_bundle import verify_modular_bundle


class JsonLineWorker:
    """Small synchronous request client for one persistent JSONL worker process."""

    def __init__(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
        startup_timeout_seconds: float = 300.0,
        request_timeout_seconds: float = 600.0,
    ) -> None:
        self.command = command
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_stream = self.log_path.open("wb")
        self._process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log_stream,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._stdout_reader = threading.Thread(
            target=self._read_stdout, name=f"worker-stdout-{self._process.pid}", daemon=True
        )
        self._stdout_reader.start()
        self.request_timeout_seconds = request_timeout_seconds
        started = time.perf_counter()
        try:
            ready = self._read_response(startup_timeout_seconds)
        except Exception:
            self.close(timeout_seconds=1.0)
            raise
        self.startup_wall_seconds = time.perf_counter() - started
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(f"worker failed during startup: {ready}")
        self.ready = ready

    def _read_stdout(self) -> None:
        if self._process.stdout is None:
            self._responses.put(None)
            return
        try:
            for line in self._process.stdout:
                self._responses.put(line)
        finally:
            self._responses.put(None)

    def _read_response(self, timeout_seconds: float | None = None) -> dict[str, Any]:
        try:
            line = self._responses.get(timeout=timeout_seconds)
        except queue.Empty as error:
            return_code = self._process.poll()
            raise TimeoutError(
                "worker response timed out "
                f"after {timeout_seconds}s (returncode={return_code}): {self._log_tail()}"
            ) from error
        if line is None:
            return_code = self._process.poll()
            log_tail = self._log_tail()
            raise RuntimeError(
                f"worker exited before a response (returncode={return_code}): {log_tail}"
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"worker emitted non-JSON protocol output: {line[:500]!r}"
            ) from error
        if not isinstance(response, dict):
            raise RuntimeError("worker response must be a JSON object")
        return response

    def _log_tail(self, limit: int = 4000) -> str:
        try:
            self._log_stream.flush()
            with self.log_path.open("rb") as stream:
                stream.seek(max(0, self.log_path.stat().st_size - limit))
                return stream.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._process.poll() is not None:
            raise RuntimeError(
                f"worker is no longer running ({self._process.returncode}): {self._log_tail()}"
            )
        if self._process.stdin is None:
            raise RuntimeError("worker stdin pipe is unavailable")
        self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        response = self._read_response(self.request_timeout_seconds)
        if response.get("error"):
            error = response["error"]
            raise RuntimeError(f"worker request failed: {error}")
        return response

    def close(self, timeout_seconds: float = 10.0) -> None:
        if self._process.poll() is None:
            try:
                if self._process.stdin is not None:
                    self._process.stdin.write('{"command":"shutdown"}\n')
                    self._process.stdin.flush()
                    self._read_response(timeout_seconds=timeout_seconds)
            except (BrokenPipeError, RuntimeError, TimeoutError, OSError):
                pass
            try:
                self._process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
        for stream in (self._process.stdin, self._process.stdout):
            if stream is not None:
                stream.close()
        if not self._log_stream.closed:
            self._log_stream.close()
        self._stdout_reader.join(timeout=1.0)


class PathARuntimeDispatcher:
    """Route typed task requests through resident Torch and VideoPrism workers.

    Request forms are task-specific and explicit:
    - text: ``state``, ``question``, ``options``
    - audio: ``audio_path``, ``queries`` (question/options pairs)
    - image: ``image_path``, ``taxonomy``, ``options``
    - video: ``video_path``, ``question``, ``candidate_statement``
    """

    def __init__(
        self,
        assembly_dir: Path,
        *,
        videoprism_python: Path,
        videoprism_source: Path,
        worker_log_dir: Path,
        device: str = "cuda",
    ) -> None:
        self.assembly_dir = assembly_dir.resolve(strict=True)
        verify_started = time.perf_counter()
        self.manifest = verify_modular_bundle(self.assembly_dir)
        self.bundle_verification_seconds = time.perf_counter() - verify_started
        self.videoprism_python = videoprism_python.resolve(strict=True)
        self.videoprism_source = videoprism_source.resolve(strict=True)
        self.worker_log_dir = worker_log_dir.resolve()
        if self.worker_log_dir.exists() and any(self.worker_log_dir.iterdir()):
            raise FileExistsError(f"worker log directory is not empty: {self.worker_log_dir}")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        self.device = device
        self._torch_worker: JsonLineWorker | None = None
        self._videoprism_worker: JsonLineWorker | None = None
        self._request_id = 0
        self._project_root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        source_paths = [str(self._project_root / "src"), str(self._project_root)]
        old_pythonpath = env.get("PYTHONPATH")
        if old_pythonpath:
            source_paths.append(old_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(source_paths)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        self._worker_env = env

    @property
    def worker_metadata(self) -> dict[str, Any]:
        return {
            "torch": self._torch_worker.ready if self._torch_worker else None,
            "videoprism": self._videoprism_worker.ready if self._videoprism_worker else None,
        }

    def _next_id(self) -> str:
        self._request_id += 1
        return f"path-a-{self._request_id:06d}"

    def _ensure_torch(self) -> JsonLineWorker:
        if self._torch_worker is None:
            log_path = self.worker_log_dir / "torch-worker.log"
            self._torch_worker = JsonLineWorker(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "scripts.path_a_torch_worker",
                    "--assembly",
                    str(self.assembly_dir),
                    "--device",
                    self.device,
                ],
                cwd=self._project_root,
                env=self._worker_env,
                log_path=log_path,
            )
        return self._torch_worker

    def _ensure_videoprism(self) -> JsonLineWorker:
        if self._videoprism_worker is None:
            log_path = self.worker_log_dir / "videoprism-worker.log"
            self._videoprism_worker = JsonLineWorker(
                [
                    str(self.videoprism_python),
                    "-B",
                    "-m",
                    "scripts.path_a_videoprism_worker",
                    "--assembly",
                    str(self.assembly_dir),
                    "--source-root",
                    str(self.videoprism_source),
                ],
                cwd=self._project_root,
                env=self._worker_env,
                log_path=log_path,
            )
        return self._videoprism_worker

    def dispatch(self, task: str, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_id()
        started = time.perf_counter()
        torch_was_running = self._torch_worker is not None
        videoprism_was_running = self._videoprism_worker is not None
        if task == "image":
            response = self._ensure_videoprism().request(
                {"request_id": request_id, "task": task, "payload": payload}
            )
        elif task == "video":
            encoded = self._ensure_videoprism().request(
                {"request_id": request_id, "task": "encode_video", "payload": payload}
            )
            scored = self._ensure_torch().request(
                {
                    "request_id": request_id,
                    "task": "video_score",
                    "payload": {
                        **payload,
                        "video_feature": encoded["video_feature"],
                    },
                }
            )
            response = {**scored, "video_encoder": encoded["encoder_metrics"]}
        elif task in {"text", "audio"}:
            response = self._ensure_torch().request(
                {"request_id": request_id, "task": task, "payload": payload}
            )
        else:
            raise ValueError(f"unsupported Path A task: {task}")
        startup_wall = {}
        if not torch_was_running and self._torch_worker is not None:
            startup_wall["torch"] = self._torch_worker.startup_wall_seconds
        if not videoprism_was_running and self._videoprism_worker is not None:
            startup_wall["videoprism"] = self._videoprism_worker.startup_wall_seconds
        return {
            **response,
            "dispatch_request_id": request_id,
            "dispatch_task": task,
            "dispatch_wall_seconds": time.perf_counter() - started,
            "worker_startup_wall_seconds": startup_wall,
        }

    def close(self) -> None:
        for worker in (self._torch_worker, self._videoprism_worker):
            if worker is not None:
                worker.close()
        self._torch_worker = None
        self._videoprism_worker = None

    def __enter__(self) -> PathARuntimeDispatcher:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
