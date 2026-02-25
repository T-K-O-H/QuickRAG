"""API key authentication middleware for QuickRAG."""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from quickrag.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str | None:
    """Verify the API key if authentication is enabled.

    When QUICKRAG_API_KEYS is not set, authentication is disabled and all
    requests are allowed through.  When it is set (comma-separated list),
    the request must include a valid ``X-API-Key`` header.

    Returns:
        The validated API key, or None if auth is disabled.
    """
    valid_keys = settings.get_api_keys()

    # No keys configured — auth disabled
    if not valid_keys:
        return None

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
        )

    if api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key
