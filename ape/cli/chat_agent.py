from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Sequence, Optional

import importlib
from loguru import logger

from ape.settings import settings
from ape.cli.context_manager import ContextManager
from ape.cli.mcp_client import MCPClient
from ape.utils import count_tokens, get_ollama_model_info
import jwt  # PyJWT
from ape.core.agent_core import AgentCore

# NOTE: prompt_toolkit is imported lazily inside __init__ to avoid mandatory
# dependency when the library is used purely as a backend package.
PromptSession = None  # will be set in __init__

# NOTE: ChatAgent extends the reusable AgentCore with streaming/printing logic
#       specific to CLI interactions (left intact for backward compatibility).
class ChatAgent(AgentCore):
    """High-level autonomous agent.

    The class is deliberately *stateless* beyond the provided `ContextManager`
    and therefore can be instantiated multiple times in the same process (for
    different user sessions).
    """

    def __init__(
        self,
        session_id: str,
        mcp_client: MCPClient,
        context_manager: ContextManager,
        *,
        agent_name: str = "APE",
        role_definition: str = "",
    ) -> None:
        """Create a new ChatAgent.

        Parameters
        ----------
        session_id:
            Unique identifier for the end-user session.
        mcp_client:
            Connected MCPClient instance.
        context_manager:
            Per-session context storage.
        agent_name:
            Human-readable identifier (e.g. "APE-1", "APE-2") used inside the
            system prompt so that multiple agents in the same Python process
            keep their identities separate.
        role_definition:
            Role definition for the agent.
        """

        super().__init__(
            session_id=session_id,
            mcp_client=mcp_client,
            context_manager=context_manager,
            agent_name=agent_name,
            role_definition=role_definition,
            context_limit=None,  # Will be set in initialize
        )

        self.model_info = {}

        # Initialise prompt session lazily (optional dependency)
        global PromptSession  # noqa: PLW0603 – assign module-level var
        try:
            if PromptSession is None:
                from prompt_toolkit import PromptSession as _PS  # type: ignore
                PromptSession = _PS
        except ImportError:
            PromptSession = None

        self.prompt: Optional["PromptSession"] = PromptSession() if PromptSession else None

    async def initialize(self):
        """Fetch model info and then initialize memory."""
        try:
            self.model_info = await get_ollama_model_info(settings.LLM_MODEL)
            
            # Use manual override if provided, otherwise detect from model
            if settings.CONTEXT_WINDOW:
                self.context_limit = settings.CONTEXT_WINDOW
            else:
                self.context_limit = self.model_info.get("context_length")
        except Exception as exc:
            logger.warning(f"Could not retrieve model info: {exc}")
            self.model_info = {}
            self.context_limit = None

        # Now that we have the context limit, we can initialize the parent
        await super().initialize()

    # ------------------------------------------------------------------
    # All helper methods (discover_capabilities, create_dynamic_system_prompt,
    # get_ollama_tools, handle_tool_calls) are now inherited from AgentCore.
    # No overrides needed other than the streaming wrapper below.

    # ------------------------------------------------------------------
    async def chat_with_llm(self, message: str, conversation: List[Dict[str, str]]):
        """Stream interaction – delegates core logic and prints chunks."""

        def _printer(chunk: str):
            print(chunk, end="", flush=True)

        try:
            ollama = importlib.import_module("ollama")  # lazy heavy import
            client = ollama.AsyncClient(host=str(settings.OLLAMA_BASE_URL))
        except Exception as exc:
            logger.warning(f"Could not connect to Ollama: {exc}")
            client = None

        resp = await super().chat_with_llm(message, conversation, stream_callback=_printer)
        if not resp.endswith("\n"):
            print()
        return resp 