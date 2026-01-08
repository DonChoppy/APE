from __future__ import annotations

"""Simple asyncio connection pool for *aiosqlite*.

This lightweight helper avoids paying the open/close penalty for every query
and keeps at most *size* file handles alive for each database file.

All callers should use the ``get_db(db_path)`` async context manager:

```python
from ape.settings import settings

async with get_db(settings.SESSION_DB_PATH) as conn:
    await conn.execute(...)
```
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import aiosqlite

class _AioSqlitePool:
    def __init__(self, db_path: str, size: int = 5):
        self.db_path = db_path
        self._size = size
        self._queue: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(maxsize=size)
        self._initialised = False

    async def _init_pool(self) -> None:
        """Open *size* connections and put them into the queue."""
        if self._initialised:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        for _ in range(self._size):
            # Increase timeout to 60s to handle high concurrency in Docker volumes
            conn = await aiosqlite.connect(self.db_path, timeout=60.0)
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")  # Optimization for WAL
            await self._queue.put(conn)
        self._initialised = True

    async def acquire(self) -> aiosqlite.Connection:
        if not self._initialised:
            await self._init_pool()
        return await self._queue.get()

    async def release(self, conn: aiosqlite.Connection) -> None:
        await self._queue.put(conn)

    async def close(self) -> None:
        while not self._queue.empty():
            conn = await self._queue.get()
            await conn.close()
        self._initialised = False

# Global dictionary to hold pools for different database paths
_POOLS: Dict[str, _AioSqlitePool] = {}
_POOLS_LOCK = asyncio.Lock()

async def get_pool(db_path: str) -> _AioSqlitePool:
    """Get or create a connection pool for a given database path."""
    async with _POOLS_LOCK:
        if db_path not in _POOLS:
            _POOLS[db_path] = _AioSqlitePool(db_path, size=5)
        return _POOLS[db_path]

async def close_all_pools() -> None:
    """Close all active connection pools."""
    async with _POOLS_LOCK:
        for pool in _POOLS.values():
            await pool.close()
        _POOLS.clear()

@asynccontextmanager
async def get_db(db_path: str):
    """Provides a connection from the pool for the specified database path."""
    pool = await get_pool(db_path)
    conn = await pool.acquire()
    try:
        yield conn
    finally:
        await pool.release(conn)