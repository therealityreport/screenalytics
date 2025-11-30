from __future__ import annotations

from typing import Any, Mapping

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def _as_envelope(code: str, message: str, details: Any | None = None) -> Mapping[str, Any]:
    """Standardize error payloads for the API and UI."""
    payload = {"code": code, "message": message}
    if details is not None:
        payload["details"] = details
    return payload


def install_error_handlers(app: FastAPI) -> None:
    """Attach consistent error handlers producing {code, message, details} envelopes."""

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        headers = exc.headers or {}
        code = headers.get("x-error-code") or f"HTTP_{exc.status_code}"
        message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        details = None if isinstance(exc.detail, str) else exc.detail
        return JSONResponse(status_code=exc.status_code, content=_as_envelope(code, message, details))

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_as_envelope("VALIDATION_ERROR", "Validation error", exc.errors()),
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover - safety net
        # Avoid leaking stack traces to clients; keep minimal detail.
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_as_envelope("INTERNAL_ERROR", "Internal server error", {"error": str(exc)}),
        )
