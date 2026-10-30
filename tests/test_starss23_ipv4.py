import socket
from unittest.mock import Mock, patch

from scripts.run_starss23_ipv4 import IPv4HTTPSConnection, ipv4_connection


def test_ipv4_transport_preserves_original_tls_hostname():
    connection = IPv4HTTPSConnection("zenodo.org")
    assert connection.host == "zenodo.org"
    assert connection._context.check_hostname
    assert connection._create_connection is ipv4_connection


def test_ipv4_resolution_and_socket_timeout_are_preserved():
    addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.1", 443))]
    sock = Mock()
    with (
        patch("socket.getaddrinfo", return_value=addresses) as resolve,
        patch("socket.create_connection", return_value=sock) as connect,
    ):
        assert ipv4_connection(("zenodo.org", 443), 10) is sock
    resolve.assert_called_once_with("zenodo.org", 443, socket.AF_INET, socket.SOCK_STREAM)
    connect.assert_called_once_with(("192.0.2.1", 443), 10, None)
