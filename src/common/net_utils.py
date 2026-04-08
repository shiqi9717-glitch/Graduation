"""Networking helpers shared by model clients and data tools."""

from typing import Any, Dict, Optional

import httpx


def extract_error_code(exc: Exception) -> str:
    cause = getattr(exc, "__cause__", None)
    if cause is None:
        return "unknown"
    errno = getattr(cause, "errno", None)
    if errno is None:
        return str(cause)
    return str(errno)


def build_async_httpx_client(
    timeout: float,
    verify_ssl: bool,
    headers: Optional[Dict[str, str]] = None,
    proxy_url: Optional[str] = None,
) -> httpx.AsyncClient:
    kwargs: Dict[str, Any] = {
        "timeout": timeout,
        "verify": verify_ssl,
    }
    if headers:
        kwargs["headers"] = headers
    if proxy_url:
        try:
            return httpx.AsyncClient(proxies=proxy_url, **kwargs)
        except TypeError:
            kwargs["proxy"] = proxy_url
            return httpx.AsyncClient(**kwargs)
    return httpx.AsyncClient(**kwargs)
