"""Process-local IPv4 transport for STARSS23; retains HTTPS hostname/TLS validation."""

from __future__ import annotations

import http.client
import socket
import urllib.request
from unittest.mock import patch


def ipv4_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
    host, port = address
    last_error = None
    for _, _, _, _, resolved in socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM):
        try:
            return socket.create_connection((resolved[0], port), timeout, source_address)
        except OSError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise OSError("no IPv4 address available")


class IPv4HTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = ipv4_connection


class IPv4HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, request):
        return self.do_open(IPv4HTTPSConnection, request, context=self._context)


def main():
    from scripts.fetch_starss23_frozen_media import main as acquire

    opener = urllib.request.build_opener(IPv4HTTPSHandler())
    # Single-purpose, single-threaded child process; no system network settings changed.
    with patch("urllib.request.urlopen", opener.open):
        acquire()


if __name__ == "__main__":
    main()
