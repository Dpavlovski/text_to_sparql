from typing import Optional

import aiohttp

_session: Optional[aiohttp.ClientSession] = None


def get_session() -> aiohttp.ClientSession:
    """
    Returns the shared aiohttp.ClientSession, creating it if it doesn't exist.
    """
    global _session
    # Create a new session if one doesn't exist or if the old one is closed.
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


async def close_session():
    """
    Closes the shared aiohttp.ClientSession if it exists and is open.
    """
    global _session
    if _session and not _session.closed:
        await _session.close()
    _session = None
