"""Authentication and security middleware."""

import secrets
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

# API Key header
api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify API key if authentication is enabled."""
    if not settings.require_api_key:
        return "no-auth"

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # In production, compare against stored hashed keys
    if not secrets.compare_digest(api_key, settings.secret_key):
        raise HTTPException(status_code=403, detail="Invalid API key")

    return api_key


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""

    def __init__(self, app, calls: int = 100, window: int = 60):
        super().__init__(app)
        self.calls = calls
        self.window = window
        self.requests = {}

    async def dispatch(self, request: Request, call_next):
        import time

        # Get client IP
        client_ip = request.client.host
        current_time = time.time()

        # Clean old entries
        self.requests = {
            ip: times
            for ip, times in self.requests.items()
            if any(t > current_time - self.window for t in times)
        }

        # Check rate limit
        if client_ip in self.requests:
            recent_requests = [
                t for t in self.requests[client_ip] if t > current_time - self.window
            ]

            if len(recent_requests) >= self.calls:
                logger.warning(f"Rate limit exceeded for {client_ip}")
                return Response(
                    content="Rate limit exceeded",
                    status_code=429,
                    headers={
                        "Retry-After": str(self.window),
                        "X-RateLimit-Limit": str(self.calls),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(current_time + self.window)),
                    },
                )

            self.requests[client_ip] = recent_requests + [current_time]
        else:
            self.requests[client_ip] = [current_time]

        # Add rate limit headers
        response = await call_next(request)
        remaining = self.calls - len(self.requests.get(client_ip, []))
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.window))

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "img-src 'self' data: https:; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "font-src 'self' data:; "
            "connect-src 'self' https:;"
        )
        response.headers["Content-Security-Policy"] = csp

        return response


def validate_file_path(path: str) -> bool:
    """Validate file path to prevent directory traversal."""
    import os
    from pathlib import Path

    # Normalize the path
    normalized = os.path.normpath(path)

    # Check for path traversal attempts
    if ".." in normalized or normalized.startswith("/"):
        return False

    # Additional checks
    forbidden_chars = ["~", "|", "&", ";", "$", "(", ")", "`", "<", ">"]
    if any(char in path for char in forbidden_chars):
        return False

    return True


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent security issues."""
    import re
    from pathlib import Path

    # Get just the filename without path
    filename = Path(filename).name

    # Remove any non-alphanumeric characters except dots, dashes, and underscores
    sanitized = re.sub(r"[^\w\s.-]", "", filename)

    # Remove leading dots
    sanitized = sanitized.lstrip(".")

    # Limit length
    max_length = 255
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[: max_length - len(ext)] + ext

    return sanitized or "unnamed"
