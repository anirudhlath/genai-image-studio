"""Tests for authentication and security."""

from fastapi.testclient import TestClient
import pytest

from windsurf_dreambooth.api.auth import (
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    sanitize_filename,
    validate_file_path,
)


class TestFileSecurity:
    """Test file path validation and sanitization."""

    def test_validate_file_path_valid(self):
        """Test valid file paths."""
        assert validate_file_path("image.jpg") is True
        assert validate_file_path("folder/image.png") is True
        assert validate_file_path("deep/nested/path/file.txt") is True

    def test_validate_file_path_invalid(self):
        """Test invalid file paths."""
        assert validate_file_path("../etc/passwd") is False
        assert validate_file_path("/etc/passwd") is False
        assert validate_file_path("path/../../sensitive") is False
        assert validate_file_path("file;rm -rf /") is False
        assert validate_file_path("file$(whoami)") is False

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        assert sanitize_filename("normal.jpg") == "normal.jpg"
        assert sanitize_filename("file with spaces.png") == "file with spaces.png"
        assert sanitize_filename("../../../etc/passwd") == "passwd"
        assert sanitize_filename("file<script>.js") == "filescript.js"
        assert sanitize_filename(".hidden") == "hidden"
        assert sanitize_filename("") == "unnamed"


class TestRateLimiting:
    """Test rate limiting middleware."""

    def test_rate_limit_allows_requests(self):
        """Test that requests within limit are allowed."""
        from fastapi import FastAPI

        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, calls=5, window=60)

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        # Make requests within limit
        for _ in range(5):
            response = client.get("/test")
            assert response.status_code == 200

    def test_rate_limit_blocks_excess(self):
        """Test that excess requests are blocked."""
        from fastapi import FastAPI

        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, calls=2, window=60)

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        # Make requests
        for i in range(3):
            response = client.get("/test")
            if i < 2:
                assert response.status_code == 200
            else:
                assert response.status_code == 429
                assert "Retry-After" in response.headers


class TestSecurityHeaders:
    """Test security headers middleware."""

    def test_security_headers_added(self):
        """Test that security headers are added to responses."""
        from fastapi import FastAPI

        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert "Content-Security-Policy" in response.headers


if __name__ == "__main__":
    pytest.main([__file__])
