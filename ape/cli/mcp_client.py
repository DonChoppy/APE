from __future__ import annotations

import asyncio
import traceback
from typing import Any, Dict, Optional

from loguru import logger
from mcp import ClientSession

# Reworked to use the MCP SDK's built-in HTTP/SSE client
# This replaces the old stdio implementation.
from mcp.client.sse import sse_client

from ape.settings import settings


class MCPClient:
    """Manage an MCP session over HTTP/SSE.
    
    Simplified: Connects to a SINGLE Core Server (Gateway), which handles
    all downstream tools.
    """

    def __init__(self):
        self._sse_context: Optional[asyncio.AbstractAsyncContextManager] = None
        self._session_context: Optional[asyncio.AbstractAsyncContextManager] = None
        self.mcp_session: Optional[ClientSession] = None

    # ---------------------------------------------------------------------
    # Connection management
    # ---------------------------------------------------------------------
    async def connect(self) -> bool:
        """Connect to the MCP server via HTTP/SSE."""
        if self.mcp_session:
            logger.debug("MCPClient.connect(): already connected – skipping")
            return True

        try:
            # The server runs on its own, so we connect to its URL
            # We use settings.MCP_SERVER_URL.
            # If that is unset, we fallback to default or error?
            # For this phase, we assume settings.MCP_SERVER_URLS might still exist but we want the PRIMARY one.
            # Or we revert settings first?
            # Let's assume we use the first URL if array, or the legacy URL.
            
            # Temporary bridge logic until settings.py is fully reverted
            url = None
            if hasattr(settings, "MCP_SERVER_URL") and settings.MCP_SERVER_URL:
                 url = str(settings.MCP_SERVER_URL)
            elif hasattr(settings, "MCP_SERVER_URLS") and settings.MCP_SERVER_URLS:
                 url = str(settings.MCP_SERVER_URLS[0])
            else:
                 url = "http://localhost:8000"

            server_url = url.rstrip("/") + "/mcp/sse"
            logger.info(f"🔗 [MCP CLIENT] Connecting to MCP server at {server_url}…")

            # create the SSE transport context by passing the URL directly
            self._sse_context = sse_client(url=server_url)
            read, write = await self._sse_context.__aenter__()
            logger.info("📡 [MCP CLIENT] SSE connection established")

            # wrap the low-level transport in the higher-level ClientSession
            self._session_context = ClientSession(read, write)
            self.mcp_session = await self._session_context.__aenter__()
            logger.info("🤝 [MCP CLIENT] MCP session created")

            # do protocol handshake
            await self.mcp_session.initialize()
            logger.info("✅ [MCP CLIENT] MCP connection initialized successfully")

            return True
        except Exception as exc:
            logger.error(f"❌ [MCP CLIENT] Failed to connect to MCP server: {exc}")
            # traceback.print_exc()
            # make sure objects are cleaned up in case of partial failure
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        """Gracefully close the MCP session and underlying SSE transport."""
        try:
            try:
                if self._session_context is not None:
                    await self._session_context.__aexit__(None, None, None)
                    self._session_context = None
                    logger.info("🤝 [MCP CLIENT] MCP session closed")

                if self._sse_context is not None:
                    await self._sse_context.__aexit__(None, None, None)
                    self._sse_context = None
                    logger.info("📡 [MCP CLIENT] SSE connection closed")
            except (RuntimeError, asyncio.CancelledError) as exc:
                logger.warning(f"Ignoring expected shutdown error: {exc}")

            self.mcp_session = None
            logger.info("✅ [MCP CLIENT] Disconnected successfully")
        except Exception as exc:
            logger.error(f"❌ [MCP CLIENT] Error when disconnecting: {exc}")

    # ------------------------------------------------------------------
    # Convenience pass-through helpers
    # ------------------------------------------------------------------
    async def list_tools(self):
        if not self.mcp_session:
            raise RuntimeError("MCPClient not connected – call connect() first")
        return await self.mcp_session.list_tools()

    async def list_prompts(self):
        if not self.mcp_session:
            raise RuntimeError("MCPClient not connected – call connect() first")
        return await self.mcp_session.list_prompts()

    async def list_resources(self):
        if not self.mcp_session:
            raise RuntimeError("MCPClient not connected – call connect() first")
        return await self.mcp_session.list_resources()

    async def call_tool(self, name: str, arguments: Dict[str, Any]):
        if not self.mcp_session:
            raise RuntimeError("MCPClient not connected – call connect() first")
        return await self.mcp_session.call_tool(name, arguments)

    # ------------------------------------------------------------------
    # Helpers / shortcuts
    # ------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self.mcp_session is not None