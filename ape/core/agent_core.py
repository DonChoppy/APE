from __future__ import annotations

import json
from datetime import datetime, timedelta
import secrets
from typing import Any, Dict, List

from loguru import logger

from ape.utils import count_tokens
from ape.settings import settings

class AgentCore:
    """Reusable core engine shared by CLI, web, and other front-ends.

    This class contains **no I/O** (printing, prompt_toolkit, etc.) so it can
    be unit-tested in isolation.  Front-ends handle user interaction and call
    `chat_with_llm()`.
    """

    def __init__(
        self,
        session_id: str,
        mcp_client,
        context_manager,
        *,
        agent_name: str = "APE",
        role_definition: str = "",
        context_limit: int | None = None,
    ) -> None:
        self.session_id = session_id
        self.mcp_client = mcp_client
        self.context_manager = context_manager
        self.agent_name = agent_name
        self.role_definition = role_definition
        self.context_limit = context_limit
        self.memory = None
        self.vector_memory = None

    async def initialize(self):
        """Initializes the memory modules."""
        # ------------------------------------------------------------------
        # M2 – Context Intelligence: initialise hybrid window memory
        # ------------------------------------------------------------------
        try:
            from ape.core.memory import WindowMemory  # local import to avoid circular deps

            # Prefer explicit override, fallback to attribute from subclass, then 8192
            if self.context_limit is not None:
                ctx_limit = self.context_limit
            else:
                ctx_limit = getattr(self, "context_limit", 8192)

            self.memory = WindowMemory(ctx_limit=ctx_limit, mcp_client=self.mcp_client, session_id=self.session_id)
        except Exception as exc:  # pragma: no cover – memory must not break agent
            logger.warning(f"WindowMemory disabled: {exc}")
            self.memory = None

    def get_agent_card(self) -> Dict[str, Any]:
        """Return a dictionary with the agent's configuration and state."""
        card = {
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "role_definition": self.role_definition,
            "context_limit": self.context_limit,
            "model_info": getattr(self, 'model_info', {}),
            "settings": {
                "LLM_MODEL": settings.LLM_MODEL,
                "SHOW_THOUGHTS": settings.SHOW_THOUGHTS,
                "TEMPERATURE": settings.TEMPERATURE,
                "TOP_P": settings.TOP_P,
                "TOP_K": settings.TOP_K,
            }
        }
        if self.memory:
            card["memory"] = {
                "class": self.memory.__class__.__name__,
                "tokens": self.memory.tokens(),
                "summary_length": len(self.memory.summary) if hasattr(self.memory, 'summary') else 0,
            }
        return card

    async def refresh_context_window(self) -> None:
        """A capability that allows the agent to clear its own short-term memory.

        This is a powerful tool for an agent that gets stuck in a repetitive
        loop or needs to switch contexts. It forces the current message buffer
        to be summarized and archived, giving the agent a 'fresh start' for its
        next reasoning cycle.
        """
        if hasattr(self, "memory") and self.memory:
            logger.info(f"Agent {self.agent_name} is refreshing its context window.")
            await self.memory.force_summarize()
        else:
            logger.warning("refresh_context_window called but no memory module is available.")

    # ------------------------------------------------------------------
    # The following methods are *verbatim* copies of ChatAgent – kept here so
    # existing behaviour stays identical.  They deliberately avoid touching
    # stdout or stdin so that any UI layer can manage display.
    # ------------------------------------------------------------------

    async def discover_capabilities(self) -> Dict[str, Any]:
        from ape.prompts import list_prompts as _local_list  # local import

        # ------------------------------------------------------------------
        # Simple per-agent cache with 5-minute TTL to avoid a round-trip on
        # every turn while still reflecting dynamic tool changes.
        # ------------------------------------------------------------------

        TTL = timedelta(minutes=5)
        now = datetime.utcnow()

        if (
            hasattr(self, "_cached_capabilities")
            and hasattr(self, "_caps_timestamp")
            and (now - self._caps_timestamp) < TTL
        ):
            return self._cached_capabilities  # type: ignore[attr-defined]

        capabilities: Dict[str, Any] = {"tools": [], "prompts": [], "resources": []}

        if not self.mcp_client.is_connected:
            logger.warning("discover_capabilities(): MCP not connected")
            return capabilities

        # Tools
        try:
            tools_result = await self.mcp_client.list_tools()
            capabilities["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                }
                for tool in tools_result.tools
            ]
        except Exception as exc:
            logger.error(f"list_tools failed: {exc}")

        # Prompts (server may not implement)
        try:
            prompts_result = await self.mcp_client.list_prompts()
            prompt_items = getattr(prompts_result, "prompts", prompts_result)
            capabilities["prompts"] = [
                {
                    "name": p.name,
                    "description": p.description,
                    "arguments": [getattr(arg, "dict", lambda: arg)() for arg in getattr(p, "arguments", [])],
                }
                for p in prompt_items
            ]
        except Exception:
            try:
                capabilities["prompts"] = [
                    {
                        "name": prm.name,
                        "description": prm.description,
                        "arguments": [arg.dict() for arg in prm.arguments],
                    }
                    for prm in _local_list()
                ]
            except Exception as e:
                logger.error(f"Failed to discover prompts: {e}")

        # Resources
        try:
            resources_result = await self.mcp_client.list_resources()
            capabilities["resources"] = [
                res.model_dump() for res in resources_result.resources
            ]
        except Exception as e:
            logger.error(f"Failed to discover resources: {e}")
        
        logger.debug(f"Capabilities resources: {capabilities['resources']}")

        # Cache result with timestamp
        self._cached_capabilities = capabilities  # type: ignore[attr-defined]
        self._caps_timestamp = now  # type: ignore[attr-defined]

        return capabilities

    async def create_dynamic_system_prompt(self, capabilities: Dict[str, Any], security_nonce: str = "") -> str:
        from ape.prompts import render_prompt  # local import

        def _fmt(items: List[Dict[str, Any]], include_uri: bool = False) -> str:
            if not items:
                return "None"

            lines: List[str] = []
            for itm in items:
                if "parameters" in itm and isinstance(itm["parameters"], dict):
                    args = ", ".join(itm["parameters"].get("properties", {}).keys())
                elif "arguments" in itm:
                    args = ", ".join(a.get("name", "?") for a in itm["arguments"])
                else:
                    args = ""
                if args:
                    args = f" (args: {args})"

                uri_part = ""
                if include_uri and "uri" in itm:
                    uri_part = f" (uri: {str(itm['uri'])})"

                lines.append(f"• **{itm['name']}**{uri_part}{args}: {itm['description']}")
            return "\n".join(lines)

        tools_section = _fmt(capabilities["tools"])
        prompts_section = _fmt(capabilities["prompts"])
        resources_section = _fmt(capabilities["resources"], include_uri=True)

        return render_prompt(
            "system",
            {
                "agent_name": self.agent_name,
                "current_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tools_section": tools_section,
                "prompts_section": prompts_section,
                "resources_section": resources_section,
                "role_definition": getattr(self, "role_definition", ""),
                # Expose WindowMemory summary inside the system prompt (M2)
                "memory_summary": getattr(self.memory, "latest_context", lambda: "")(),
                "security_nonce": security_nonce,
            },
        )

    async def get_ollama_tools(self) -> List[Dict[str, Any]]:
        if not self.mcp_client.is_connected:
            return []
        tools_result = await self.mcp_client.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema,
                },
            }
            for t in tools_result.tools
        ]

    # ------------------------------------------------------------------
    async def handle_tool_calls(self, tool_calls: List[Dict[str, Any]], security_nonce: str = ""):
        """Execute tool calls and return formatted tool output string."""

        if not self.mcp_client.is_connected:
            return "❌ MCP client is not connected."

        results: List[Dict[str, Any]] = []

        for call in tool_calls:
            fn = call["function"]["name"]
            arguments = call["function"]["arguments"]

            # --------------------------------------------------------------
            # Rate-limiting (per session) – block excessive tool spam
            # --------------------------------------------------------------
            try:
                from ape.core.rate_limiter import allow as _rl_allow

                if not _rl_allow(self.session_id):
                    results.append({
                        "tool": fn,
                        "arguments": arguments,
                        "result": "RATE_LIMIT_EXCEEDED: Too many tool calls per minute; slow down.",
                    })
                    # Skip actual execution for this tool call
                    continue
            except Exception as exc:  # pragma: no cover – limiter must not crash tool flow
                logger.debug(f"Rate-limiter check failed (ignored): {exc}")

            # Simple placeholder substitution using the context manager
            if isinstance(arguments, dict):
                for k, v in list(arguments.items()):
                    if (
                        isinstance(v, str)
                        and v == "retrieved_session_id"
                        and "last_session_id" in self.context_manager.extracted_values
                    ):
                        arguments[k] = self.context_manager.extracted_values["last_session_id"]

            logger.bind(agent=self.agent_name).info(
                f"Executing tool {fn} with args {arguments}")
            try:
                res = await self.mcp_client.call_tool(fn, arguments)
                raw = res.content[0].text if res.content else ""

                # verify JWT
                verified = False
                payload_text = ""
                verification_error = ""
                try:
                    env = json.loads(raw)
                    token = env.get("jwt") or env.get("sig") or ""
                    if token:
                        import jwt
                        try:
                            decoded = jwt.decode(token, settings.MCP_JWT_KEY, algorithms=["HS256"])
                            verified = True
                            payload_text = decoded.get("payload") or json.dumps(decoded, ensure_ascii=False)
                        except jwt.ExpiredSignatureError as sig_exc:
                            verification_error = f"Signature expired: {sig_exc}"
                        except jwt.InvalidTokenError as sig_exc:
                            verification_error = f"Invalid signature: {sig_exc}"
                    else:
                        verification_error = "Missing JWT signature in tool result."
                        payload_text = env.get("payload", "") or raw
                except Exception as exc_inner:
                    verification_error = f"Malformed tool response: {exc_inner}"
                    payload_text = raw

                if verified:
                    text = payload_text
                else:
                    # If the tool wrapper returned a plain error string (no JWT), surface it.
                    if payload_text:
                        text = f"❌ TOOL ERROR: {payload_text.strip()}"
                    else:
                        text = f"❌ ERROR: Tool result signature verification failed – {verification_error or 'unknown reason.'}"
                # Log signature verification failure as structured error
                if not verified:
                    try:
                        from ape.mcp.session_manager import get_session_manager

                        await get_session_manager().a_save_error(fn, arguments, "Signature verification failed", session_id=self.session_id)
                    except Exception as log_exc:  # pragma: no cover – logging must not break tool flow
                        logger.debug(f"Could not persist verification error: {log_exc}")
            except Exception as exc:
                text = f"ERROR executing tool: {exc}"
                try:
                    from ape.mcp.session_manager import get_session_manager

                    await get_session_manager().a_save_error(fn, arguments, str(exc), session_id=self.session_id)
                except Exception as log_exc:  # pragma: no cover
                    logger.debug(f"Could not persist tool error: {log_exc}")

            # --------------------------------------------------------------
            # Proactive Summarization: check if this specific result is too large
            # --------------------------------------------------------------
            try:
                from ape.utils import count_tokens
                ctx_limit = getattr(self, "context_limit", None) or 8192
                # If result is > 20% of context window or > 2000 tokens, summarize it
                threshold = min(2000, ctx_limit // 5)
                
                if count_tokens(text) > threshold and fn != "summarize_text":
                    logger.info(f"Tool result from '{fn}' is too large ({count_tokens(text)} tokens). Triggering proactive summarization.")
                    
                    try:
                        # Use internal memory's summarization if available, else direct tool call
                        if hasattr(self, "memory") and self.memory:
                            summary = await self.memory.summarize(text)
                        else:
                            res_sum = await self.mcp_client.call_tool("summarize_text", {"text": text})
                            summary = res_sum.content[0].text if res_sum.content else ""
                            # Handle potential JWT envelope in summary tool output
                            try:
                                env_sum = json.loads(summary)
                                payload_sum = env_sum.get("payload", summary)
                                tr_sum = json.loads(payload_sum)
                                summary = tr_sum.get("result", payload_sum)
                            except:
                                pass

                        if summary:
                            text = (
                                f"⚠️ [AUTO-SUMMARIZED DUE TO SIZE]\n"
                                f"The original output was too large for the context window. "
                                f"Here is a concise summary of the result:\n\n"
                                f"{summary}\n\n"
                                f"NOTE: Full data is persisted in session history if needed."
                            )
                    except Exception as sum_exc:
                        logger.warning(f"Proactive summarization failed: {sum_exc}")
            except Exception as check_exc:
                logger.debug(f"Summarization check failed: {check_exc}")

            results.append({"tool": fn, "arguments": arguments, "result": text})
            self.context_manager.add_tool_result(fn, arguments, text)


        formatted_lines = [f"🔧 SYSTEM NOTE: BEGIN_TOOL_OUTPUT (nonce: {security_nonce})\n"]
        for idx, r in enumerate(results, 1):
            tool_block = (
                f"<tool_output index=\"{idx}\" name=\"{r['tool']}\" nonce=\"{security_nonce}\">\n"
                f"Arguments: `{json.dumps(r['arguments'], ensure_ascii=False)}`\n\n"
                f"{r['result']}\n"
                f"</tool_output>\n"
            )
            formatted_lines.append(tool_block)
        formatted_lines.append(f"🔧 SYSTEM NOTE: END_TOOL_OUTPUT (nonce: {security_nonce})\n")

        return "".join(formatted_lines)

    # ------------------------------------------------------------------
    async def chat_with_llm(
        self,
        message: str,
        conversation: List[Dict[str, str]],
        *,
        stream_callback=None,
    ) -> str:
        """Run LLM chat; optional *stream_callback* receives incremental text."""
        
        # 1. Generate a cryptographic nonce for this specific interaction cycle
        # This prevents "prompt injection" from tools because injected text cannot guess this nonce.
        security_nonce = secrets.token_hex(4)

        capabilities = await self.discover_capabilities()
        system_prompt = await self.create_dynamic_system_prompt(capabilities, security_nonce=security_nonce)

        # ------------------------------------------------------------------
        # Update memory with the incoming *user* message and prune if needed
        # BEFORE sending the payload to the model so the context stays within
        # the token budget.
        # ------------------------------------------------------------------
        if hasattr(self, "memory") and self.memory:
            # Add the new user message
            self.memory.add({"role": "user", "content": message})
            await self.memory.prune()

        ctx_summary = self.context_manager.get_context_summary()
        if ctx_summary.strip() != "CURRENT SESSION CONTEXT:":
            system_prompt += f"\n\nCURRENT CONTEXT:\n{ctx_summary}"

        # If memory is managed internally, we prefer self.memory.messages over the passed conversation
        # to ensure summarization/pruning logic is honored consistently.
        if hasattr(self, "memory") and self.memory and self.memory.messages:
            history_to_use = self.memory.messages
        else:
            history_to_use = conversation

        exec_conversation = [
            {"role": "system", "content": system_prompt},
            *history_to_use,
            {"role": "user", "content": message},
        ]

        try:
            from ape.utils import count_tokens  # local import to avoid heavy deps outside use
        except Exception:
            # Fallback if count_tokens is not available
            def count_tokens(text: str) -> int:
                return len(text) // 4 # rough estimate

        try:
            tools_tokens = count_tokens(json.dumps(capabilities["tools"]))
        except Exception:
            tools_tokens = 0

        ctx_limit = getattr(self, "context_limit", None) or 8192

        def _prune(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
            """Trim middle of conversation to fit budget while keeping system + latest prompt."""
            margin = settings.CONTEXT_MARGIN_TOKENS
            total = sum(count_tokens(m["content"]) for m in messages) + tools_tokens
            
            if total <= ctx_limit - margin:
                return messages
            
            logger.info(f"Pruning conversation: {total} tokens (limit {ctx_limit}, margin {margin})")
            # Keep system prompt [0] and latest user/tool output [-1]
            sys_msg = messages[0]
            latest_msg = messages[-1]
            history = messages[1:-1]
            
            while history and total > ctx_limit - margin:
                removed = history.pop(0)
                total -= count_tokens(removed["content"])
            
            return [sys_msg, *history, latest_msg]

        import ollama
        client = ollama.AsyncClient(host=str(settings.OLLAMA_BASE_URL))
        # Normalised tools payload (OpenAI spec) – avoids 500 JSON errors
        tools_spec = await self.get_ollama_tools()

        ollama_options = {
            "temperature": settings.TEMPERATURE,
            "top_p": settings.TOP_P,
            "top_k": settings.TOP_K,
        }
        if self.context_limit:
            ollama_options["num_ctx"] = self.context_limit

        max_iter = settings.MAX_TOOLS_ITERATIONS
        iteration = 0
        cumulative_resp = ""

        while iteration < max_iter:
            current_chunk = ""
            has_tool_calls = False

            # Dynamic pruning: ensure in-flight context (including new tool results) fits budget
            exec_conversation = _prune(exec_conversation)

            # Prepare chat arguments
            chat_args = {
                "model": settings.LLM_MODEL,
                "messages": exec_conversation,
                "options": ollama_options,
                "stream": True,
            }
            if tools_spec:
                chat_args["tools"] = tools_spec
            
            # Special case for 'think' parameter due to Ollama quirks
            if not settings.SHOW_THOUGHTS:
                chat_args["think"] = False

            try:
                stream = await client.chat(**chat_args)
            except Exception as first_exc:
                if tools_spec:
                    logger.warning(f"Ollama chat failed with tools payload (retrying without tools): {first_exc}")
                    chat_args.pop("tools")
                    stream = await client.chat(**chat_args)
                else:
                    raise first_exc


            async for chunk in stream:
                if thinking := chunk.get("thinking"):
                    if stream_callback:
                        stream_callback(thinking)

                if "message" not in chunk:
                    continue
                msg = chunk["message"]
                if content := msg.get("content"):
                    if stream_callback:
                        stream_callback(content)
                    current_chunk += content

                # ------------------------------------------------------------------
                # Tool calling – handle both plural "tool_calls" (OpenAI/Qwen spec)
                # and singular "tool_call" (some models emit only one object).
                # ------------------------------------------------------------------
                tool_calls_payload = msg.get("tool_calls") or msg.get("tool_call")
                if tool_calls_payload:
                    has_tool_calls = True
                    iteration += 1

                    if current_chunk:
                        exec_conversation.append({"role": "assistant", "content": current_chunk})
                        cumulative_resp += current_chunk + "\n"
                        current_chunk = ""

                    # Convert single dict to list for downstream handler API
                    if not isinstance(tool_calls_payload, list):
                        tool_calls_payload = [tool_calls_payload]

                    tool_result_str = await self.handle_tool_calls(tool_calls_payload, security_nonce=security_nonce)
                    if stream_callback:
                        stream_callback("\n" + tool_result_str)
                    exec_conversation.append({"role": "tool", "content": tool_result_str})
                    break

            if not has_tool_calls:
                if current_chunk:
                    cumulative_resp += current_chunk

                # Store assistant answer in WindowMemory (if enabled)
                if hasattr(self, "memory") and self.memory:
                    self.memory.add({"role": "assistant", "content": cumulative_resp})

                break

        return cumulative_resp 