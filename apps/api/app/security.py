"""HTTP hardening: body size limit, cross-site request blocking and security headers.

These are plain ASGI middlewares so they run before any request body is parsed.
"""

import json

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")


def _header(scope: Scope, name: bytes) -> bytes | None:
    for key, value in scope.get("headers", []):
        if key == name:
            return value
    return None


async def _send_json(send: Send, status_code: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class RequestTooLarge(HTTPException):
    # An HTTPException, so FastAPI re-raises it as-is when it surfaces during body parsing.
    def __init__(self, max_bytes: int):
        super().__init__(413, "Request body is too large")
        self.max_bytes = max_bytes


async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """422 without echoing the submitted values.

    FastAPI's default response includes each offending input; a NaN in the body would make
    that response itself unserializable (500), and echoing input back is unnecessary.
    """
    errors = [
        {"loc": list(err.get("loc", ())), "msg": err.get("msg", ""), "type": err.get("type", "")}
        for err in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


class BodySizeLimitMiddleware:
    """Reject request bodies larger than ``max_bytes``.

    The declared Content-Length is checked up front; chunked bodies are counted while
    they stream in, so an oversized upload is cut off instead of being spooled to disk.
    """

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _header(scope, b"content-length")
        if declared is not None:
            try:
                size = int(declared)
            except ValueError:
                await _send_json(send, status.HTTP_400_BAD_REQUEST, "Invalid Content-Length")
                return
            if size > self.max_bytes:
                exc = RequestTooLarge(self.max_bytes)
                await _send_json(send, exc.status_code, exc.detail)
                return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestTooLarge(self.max_bytes)
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except RequestTooLarge as exc:
            if response_started:
                raise
            await _send_json(send, exc.status_code, exc.detail)


class OriginCheckMiddleware:
    """Block state-changing requests sent by other websites (CSRF).

    A multipart form POST is a CORS "simple request": browsers send it without a
    preflight, so CORS alone does not stop a foreign page from uploading files or
    starting jobs. Browsers always attach an Origin header to such requests, so any
    unsafe request whose Origin is not allow-listed is refused. Requests without an
    Origin header (curl, scripts, server-to-server) are not affected.
    """

    def __init__(self, app: ASGIApp, allowed_origins: list[str]):
        self.app = app
        self.allowed = {origin.rstrip("/") for origin in allowed_origins}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] in UNSAFE_METHODS:
            origin = _header(scope, b"origin")
            if origin is not None and origin.decode("latin-1") not in self.allowed:
                await _send_json(send, status.HTTP_403_FORBIDDEN, "Cross-origin request blocked")
                return
        await self.app(scope, receive, send)


def _vary_on_origin(headers: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    """Ensure ``Vary: Origin`` on every response.

    CORS headers are only added when a request carries an Origin. Without Vary, a
    browser caches e.g. the <audio> element's (no Origin) response to /audio and reuses
    it for the waveform's CORS fetch of the same URL, which then fails.
    """
    for i, (key, value) in enumerate(headers):
        if key.lower() == b"vary":
            if b"origin" not in value.lower():
                headers[i] = (key, value + b", Origin")
            return headers
    return [*headers, (b"vary", b"Origin")]


class SecurityHeadersMiddleware:
    """Add defensive headers; the API serves data and files, never pages."""

    HEADERS = [
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"referrer-policy", b"no-referrer"),
    ]
    CSP = (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'")

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # The interactive docs load scripts and styles, so they keep their default policy.
        extra = self.HEADERS if scope["path"].startswith(DOCS_PATHS) else [*self.HEADERS, self.CSP]

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = _vary_on_origin([*message.get("headers", []), *extra])
            await send(message)

        await self.app(scope, receive, send_with_headers)
