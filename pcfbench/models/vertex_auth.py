"""Vertex MaaS OpenAI-compatible auth.

Vertex AI exposes DeepSeek (and other open models) at an OpenAI-compatible
endpoint that requires a fresh GCP bearer token in the Authorization
header. We refresh the token with a cache-and-expiry policy so concurrent
agent runs don't all hit ``credentials.refresh()`` at once.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Generator

import google.auth
import google.auth.transport.requests
import httpx

_REFRESH_LEEWAY_SECONDS = 60.0
_DEFAULT_TTL_SECONDS = 3000.0


class _VertexTokenSource:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    def _refresh(self) -> None:
        request = google.auth.transport.requests.Request()
        self._credentials.refresh(request)
        token = self._credentials.token
        if not isinstance(token, str):
            raise RuntimeError("Failed to obtain GCP access token")
        self._token = token
        if self._credentials.expiry is not None:
            self._expires_at = self._credentials.expiry.timestamp()
        else:
            self._expires_at = time.time() + _DEFAULT_TTL_SECONDS

    def get(self) -> str:
        with self._lock:
            if (
                self._token is None
                or time.time() + _REFRESH_LEEWAY_SECONDS >= self._expires_at
            ):
                self._refresh()
            assert self._token is not None
            return self._token


_TOKEN_SOURCE: _VertexTokenSource | None = None


def _token_source() -> _VertexTokenSource:
    global _TOKEN_SOURCE
    if _TOKEN_SOURCE is None:
        _TOKEN_SOURCE = _VertexTokenSource()
    return _TOKEN_SOURCE


class VertexBearerAuth(httpx.Auth):
    """Refreshing-bearer-token auth for Vertex MaaS OpenAI-compatible endpoint."""

    requires_request_body = False
    requires_response_body = False

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        token = _token_source().get()
        request.headers["Authorization"] = f"Bearer {token}"
        yield request


def vertex_project_id() -> str:
    project = (
        os.environ.get("VERTEX_PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT")
    )
    if not project:
        _, default_project = google.auth.default()
        project = default_project
    if not project:
        raise RuntimeError("Could not determine Vertex project; set VERTEX_PROJECT_ID.")
    return project


def vertex_maas_base_url(*, location: str = "global") -> str:
    """OpenAI-compatible chat-completions root for Vertex MaaS open models."""
    project = vertex_project_id()
    return (
        f"https://aiplatform.googleapis.com/v1/projects/{project}"
        f"/locations/{location}/endpoints/openapi"
    )
