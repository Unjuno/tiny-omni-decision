"""Run the Speech Commands extractor through Python requests' working HTTP route."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import requests

_THREAD_LOCAL = threading.local()


def _session() -> requests.Session:
    current = getattr(_THREAD_LOCAL, "session", None)
    if current is None:
        current = requests.Session()
        _THREAD_LOCAL.session = current
    return current


class RangeProxyHandler(BaseHTTPRequestHandler):
    total_bytes = 0

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:
        if self.path != "/range":
            self.send_error(404)
            return
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            request = json.loads(body)
            url = str(request["url"])
            parsed = urlparse(url)
            if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
                self.send_error(400, "only pinned Hugging Face dataset URLs are allowed")
                return
            method = str(request.get("method", "GET")).upper()
            if method not in {"HEAD", "GET"}:
                self.send_error(405)
                return
            headers = {str(key): str(value) for key, value in request.get("headers", {}).items()}
            headers["Accept-Encoding"] = "identity"
            response = None
            last_error = None
            for attempt in range(8):
                try:
                    response = _session().request(
                        method,
                        url,
                        headers=headers,
                        timeout=(15, 90),
                        allow_redirects=True,
                        stream=False,
                    )
                    if response.status_code == 429 or response.status_code >= 500:
                        delay = min(2**attempt, 30)
                        response.close()
                        time.sleep(delay)
                        continue
                    final_host = urlparse(response.url).hostname or ""
                    if final_host != "huggingface.co" and not final_host.endswith(".hf.co"):
                        response.close()
                        raise ValueError(f"unexpected Hugging Face redirect host: {final_host}")
                    if (
                        "Range" in headers
                        and response.status_code == 200
                        and len(response.content) > 8 * 1024 * 1024
                    ):
                        response.close()
                        raise ValueError(
                            "upstream ignored Range and attempted a full-file response"
                        )
                    break
                except (requests.RequestException, ValueError) as error:
                    last_error = error
                    if isinstance(error, ValueError) or attempt == 7:
                        raise
                    time.sleep(min(2**attempt, 30))
            if response is None:
                raise RuntimeError(f"upstream retry budget exhausted: {last_error}")
            payload = b"" if method == "HEAD" else response.content
            RangeProxyHandler.total_bytes += len(payload)
            self.send_response(response.status_code)
            for name in (
                "Content-Length",
                "Content-Range",
                "Accept-Ranges",
                "ETag",
                "Content-Type",
                "Last-Modified",
            ):
                value = response.headers.get(name)
                if value is not None:
                    self.send_header(name, value)
            self.end_headers()
            if payload:
                self.wfile.write(payload)
            response.close()
        except Exception as error:  # noqa: BLE001 - relay bounded upstream failure to the caller
            payload = str(error).encode("utf-8", errors="replace")[:2000]
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--hyparquet-entry", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-per-class", type=int, default=128)
    parser.add_argument("--validation-per-class", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = _args()
    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeProxyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        proxy = f"http://127.0.0.1:{server.server_port}"
        command = [
            args.node,
            args.extractor,
            "--hyparquet-entry",
            args.hyparquet_entry,
            "--range-proxy",
            proxy,
            "--output-dir",
            args.output_dir,
            "--seed",
            str(args.seed),
            "--train-per-class",
            str(args.train_per_class),
            "--validation-per-class",
            str(args.validation_per_class),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode:
            raise SystemExit(result.returncode)
        print(f"python_requests_range_bytes={RangeProxyHandler.total_bytes}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
