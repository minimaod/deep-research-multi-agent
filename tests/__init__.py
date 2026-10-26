"""All tests are offline, including accidental provider SDK calls."""
from unittest.mock import patch
import inspect
import socket

_original_connect = socket.socket.connect


def _offline_connect(sock, address):
    # Windows asyncio implements its own wakeup pipe with a localhost socketpair.
    # Permit that internal primitive, not ordinary HTTP/DB localhost connections.
    if any(frame.function == "_fallback_socketpair" for frame in inspect.stack()):
        return _original_connect(sock, address)
    raise AssertionError("External network is forbidden in offline tests")


_network_guard = patch("socket.socket.connect", new=_offline_connect)
_network_guard.start()
_tracing_guard = patch.dict("os.environ", {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"})
_tracing_guard.start()
