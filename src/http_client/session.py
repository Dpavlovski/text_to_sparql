from typing import Optional

import aiohttp

_session: Optional[aiohttp.ClientSession] = None


async def get_session() -> aiohttp.ClientSession:
    global _session
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
