"""Tests for complete A2A error code mapping (-32001 through -32005).

Covers:
  - All 5 A2A error types in core.errors
  - ErrorMappingMiddleware mapping for each code
  - HTTP status fallback mappings
  - Connection-error fallback
  - Unknown errors pass through unchanged
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.anyio

from agentique.bridge.middleware import ErrorMappingMiddleware
from agentique.core.errors import (
    AgentiqueError,
    ContentTypeNotSupportedError,
    PushNotificationNotSupportedError,
    TaskNotCancelableError,
    TaskNotFoundError,
    UnsupportedOperationError,
)


# ---------------------------------------------------------------------------
# Error type tests
# ---------------------------------------------------------------------------


def test_task_not_found_error_codes():
    err = TaskNotFoundError("not found")
    assert err.mcp_code == -32602
    assert err.a2a_code == -32001


def test_content_type_not_supported_error_codes():
    err = ContentTypeNotSupportedError("bad content type")
    assert err.mcp_code == -32600
    assert err.a2a_code == -32002
    assert isinstance(err, AgentiqueError)


def test_unsupported_operation_error_codes():
    err = UnsupportedOperationError("not supported")
    assert err.mcp_code == -32601
    assert err.a2a_code == -32003
    assert isinstance(err, AgentiqueError)


def test_task_not_cancelable_error_codes():
    err = TaskNotCancelableError("cannot cancel")
    assert err.mcp_code == -32603
    assert err.a2a_code == -32004
    assert isinstance(err, AgentiqueError)


def test_push_notification_not_supported_error_codes():
    err = PushNotificationNotSupportedError("no push")
    assert err.mcp_code == -32601
    assert err.a2a_code == -32005
    assert isinstance(err, AgentiqueError)


# ---------------------------------------------------------------------------
# ErrorMappingMiddleware map
# ---------------------------------------------------------------------------


class _FakeA2AError(Exception):
    def __init__(self, msg: str, code: int) -> None:
        super().__init__(msg)
        self.code = code


class _FakeHTTPError(Exception):
    def __init__(self, msg: str, status_code: int) -> None:
        super().__init__(msg)
        self.status_code = status_code


def _make_mw() -> ErrorMappingMiddleware:
    return ErrorMappingMiddleware()


def test_middleware_maps_32001():
    mw = _make_mw()
    exc = _FakeA2AError("task missing", -32001)
    result = mw._map_error(exc)
    assert isinstance(result, TaskNotFoundError)


def test_middleware_maps_32002():
    mw = _make_mw()
    exc = _FakeA2AError("bad content type", -32002)
    result = mw._map_error(exc)
    assert isinstance(result, ContentTypeNotSupportedError)


def test_middleware_maps_32003():
    mw = _make_mw()
    exc = _FakeA2AError("unsupported op", -32003)
    result = mw._map_error(exc)
    assert isinstance(result, UnsupportedOperationError)


def test_middleware_maps_32004():
    mw = _make_mw()
    exc = _FakeA2AError("not cancelable", -32004)
    result = mw._map_error(exc)
    assert isinstance(result, TaskNotCancelableError)


def test_middleware_maps_32005():
    mw = _make_mw()
    exc = _FakeA2AError("no push support", -32005)
    result = mw._map_error(exc)
    assert isinstance(result, PushNotificationNotSupportedError)


def test_middleware_unknown_code_passes_through():
    mw = _make_mw()
    exc = _FakeA2AError("unknown", -99999)
    result = mw._map_error(exc)
    assert result is exc


def test_middleware_http_404():
    from agentique.core.errors import AgentNotFoundError
    mw = _make_mw()
    exc = _FakeHTTPError("not found", 404)
    result = mw._map_error(exc)
    assert isinstance(result, AgentNotFoundError)


def test_middleware_http_503():
    from agentique.core.errors import AgentUnavailableError
    mw = _make_mw()
    exc = _FakeHTTPError("unavailable", 503)
    result = mw._map_error(exc)
    assert isinstance(result, AgentUnavailableError)


def test_middleware_connection_error():
    from agentique.core.errors import AgentUnavailableError
    mw = _make_mw()

    class ConnectError(Exception):
        pass

    result = mw._map_error(ConnectError("connection refused"))
    assert isinstance(result, AgentUnavailableError)


def test_middleware_plain_exception_passes_through():
    mw = _make_mw()
    exc = ValueError("something else")
    result = mw._map_error(exc)
    assert result is exc


# ---------------------------------------------------------------------------
# ErrorMappingMiddleware.process (async)
# ---------------------------------------------------------------------------


async def test_middleware_process_maps_error():
    mw = _make_mw()
    exc = _FakeA2AError("push not supported", -32005)

    async def bad_handler(req):
        raise exc

    with pytest.raises(PushNotificationNotSupportedError):
        await mw.process({}, bad_handler)


async def test_middleware_process_pass_through_on_success():
    mw = _make_mw()

    async def good_handler(req):
        return "ok"

    result = await mw.process({}, good_handler)
    assert result == "ok"


# ---------------------------------------------------------------------------
# A2A_ERROR_MAP completeness
# ---------------------------------------------------------------------------


def test_error_map_has_all_five_codes():
    mw = _make_mw()
    for code in [-32001, -32002, -32003, -32004, -32005]:
        assert code in mw.A2A_ERROR_MAP, f"Missing A2A error code {code}"
