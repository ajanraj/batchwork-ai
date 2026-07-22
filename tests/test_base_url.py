import pytest

from batchwork._base_url import BaseUrlError, normalize_base_url


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("https://gateway.example.com/v1///", "https://gateway.example.com/v1"),
        ("http://localhost:8080/v1/", "http://localhost:8080/v1"),
        ("http://127.0.0.2:8080/v1", "http://127.0.0.2:8080/v1"),
        ("http://[::1]:8080/v1/", "http://[::1]:8080/v1"),
    ),
)
def test_normalize_base_url_accepts_https_and_loopback_http(value: str, expected: str) -> None:
    assert normalize_base_url(value) == expected


@pytest.mark.parametrize(
    "value",
    (
        "http://example.com/v1",
        "http://10.0.0.1/v1",
        "http://169.254.169.254/latest",
        "http://0.0.0.0/v1",
        "https://user:secret@example.com/v1",
        "https://example.com/v1?",
        "https://example.com/v1#",
        "https://example.com/v1?region=us",
        "https://example.com/v1#fragment",
        "https://example.com:not-a-port/v1",
        "https://exa\tmple.com/v1",
        "https://exa\nmple.com/v1",
        "//example.com/v1",
    ),
)
def test_normalize_base_url_rejects_unsafe_endpoints(value: str) -> None:
    with pytest.raises(BaseUrlError):
        normalize_base_url(value)
