"""Main MCP server for APE (Advanced Prompt Engine)."""

import json
import asyncio
import time
from typing import Any, Sequence
from uuid import uuid4
import os

import mcp.types as types
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
from mcp import ClientSession
from mcp.client.sse import sse_client
from pathlib import Path
from contextlib import AsyncExitStack

from loguru import logger
from ape.utils import setup_logger

from .plugin import discover
from . import implementations_builtin
from .session_manager import get_session_manager
from ape.mcp.models import ErrorEnvelope, ToolCall, ToolResult
from ape.prompts import list_prompts as _list_prompts, render_prompt
from ape.resources import list_resources as _list_resources, read_resource as _read_resource
from ape.core.vector_memory import get_vector_memory

from ape.settings import settings  # local import to avoid circular deps

import jwt  # PyJWT
from ape.errors import ApeError  # local import


def create_mcp_server(downstream_tools: list[types.Tool] = None, downstream_sessions: dict[str, ClientSession] = None) -> Server:
    """Create and configure the MCP server with all tools and resources."""
    
    # Initialize the MCP server using the official SDK
    server = Server("ape-server")
    registry = discover()
    
    if downstream_tools is None:
        downstream_tools = []
    if downstream_sessions is None:
        downstream_sessions = {}

    SECRET = settings.MCP_JWT_KEY  # str

    def _encode_token(data: dict) -> str:
        """Return HS256-signed JWT containing *data* plus issued-at timestamp."""
        now = int(time.time())
        payload = {
            **data,
            "iat": now,
            "exp": now + 600,  # token valid for 10 minutes
        }
        return jwt.encode(payload, SECRET, algorithm="HS256")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """List available tools (local + downstream)."""
        local_tools = [
            types.Tool(name=name, description=meta["description"], inputSchema=meta["inputSchema"])
            for name, meta in registry.items()
        ]
        return local_tools + downstream_tools

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict | None
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        """Handle tool calls."""
        if arguments is None:
            arguments = {}
        
        logger.info(f"🔧 [MCP SERVER] Tool called: {name} with arguments: {arguments}")
        
        try:
            if name not in registry:
                # ------------------------------------------------------------------
                # PROXY LOGIC: Check downstream sessions
                # ------------------------------------------------------------------
                target_session = downstream_sessions.get(name)

                if target_session:
                    logger.info(f"🔗 [MCP GATEWAY] Proxying tool '{name}' to downstream.")
                    try:
                        # Proxy the call
                        result = await target_session.call_tool(name, arguments)
                        
                        # Sign the result if it's text content, so the Agent trusts it
                        if result.content and isinstance(result.content[0], types.TextContent):
                            raw_content = result.content[0].text
                            try:
                                result_data = json.loads(raw_content)
                            except:
                                result_data = raw_content
                            
                            payload_str = ToolResult(
                                tool=name,
                                arguments=arguments,
                                result=result_data,
                            ).model_dump_json()
                            rid = str(uuid4())
                            envelope = {
                                "result_id": rid,
                                "payload": payload_str,
                                "sig": _encode_token({"result_id": rid, "payload": payload_str}),
                            }
                            return [types.TextContent(type="text", text=json.dumps(envelope))]

                        return result.content
                    except Exception as e:
                        logger.error(f"❌ [MCP GATEWAY] Downstream error for '{name}': {e}")
                        return [types.TextContent(type="text", text=f"Error executing downstream tool '{name}': {e}")]
                
                # If truly not found locally or remotely
                error_msg = f"Tool '{name}' not found."
                logger.error(error_msg)
                envelope = ErrorEnvelope(error=error_msg, tool=name, request=ToolCall(name=name, arguments=arguments))
                await get_session_manager().a_save_error(name, arguments, error_msg, session_id=arguments.get("session_id"))
                return [types.TextContent(type="text", text=envelope.model_dump_json())]

            # ------------------------------------------------------------------
            # LOCAL EXECUTION LOGIC
            # ------------------------------------------------------------------

            # Sanitize arguments: drop keys not declared in the JSON schema
            schema_props = registry[name]["inputSchema"].get("properties", {})
            if schema_props:
                logger.info(f"Sanitizing arguments: {arguments} against schema: {schema_props}")
                arguments = {k: v for k, v in arguments.items() if k in schema_props}
                logger.info(f"Sanitized arguments: {arguments}")
            else:
                # No properties defined → tool expects zero arguments
                arguments = {}

            impl_fn = registry[name]["fn"]
            result_from_impl = await impl_fn(**arguments)

            try:
                # Try to parse it as JSON, so it gets embedded as an object/array
                result_data = json.loads(result_from_impl)
            except (json.JSONDecodeError, TypeError):
                # If it's not JSON, treat it as a plain string
                result_data = result_from_impl

            # Wrap successful result in a ToolResult and HMAC-signed envelope
            payload_str = ToolResult(
                tool=name,
                arguments=arguments,
                result=result_data,
            ).model_dump_json()
            rid = str(uuid4())
            envelope = {
                "result_id": rid,
                "payload": payload_str,
                "sig": _encode_token({"result_id": rid, "payload": payload_str}),
            }

            return [types.TextContent(type="text", text=json.dumps(envelope))]

        except Exception as e:
            logger.error(f"💥 [MCP SERVER] Error handling tool {name}: {e}")

            if isinstance(e, ApeError):
                err_payload = e.to_dict()
            else:
                err_payload = {"status": "error", "code": "UNHANDLED_EXCEPTION", "message": str(e)}

            envelope = ErrorEnvelope(error=json.dumps(err_payload), tool=name, request=ToolCall(tool=name, arguments=arguments))
            await get_session_manager().a_save_error(name, arguments, err_payload.get("message", str(e)), session_id=arguments.get("session_id"))
            return [types.TextContent(type="text", text=envelope.model_dump_json())]

    @server.list_prompts()
    async def handle_list_prompts() -> list[Any]:
        """Expose all prompts loaded from ``ape/prompts`` to the agent."""
        PromptModel = getattr(types, "Prompt", dict)  # type: ignore[var-annotated]
        ArgModel = (
            getattr(types, "PromptArgument", None)
            or getattr(types, "Argument", None)
            or dict
        )  # type: ignore[var-annotated]

        prompt_objs: list[Any] = []
        for p in _list_prompts():
            try:
                if PromptModel is dict:
                    prompt_objs.append(p.dict())
                else:
                    prompt_objs.append(
                        PromptModel(
                            name=p.name,
                            description=p.description,
                            arguments=[
                                (
                                    ArgModel(
                                        name=a.name,
                                        description=a.description,
                                        required=a.required,
                                    )
                                    if ArgModel is not dict
                                    else {
                                        "name": a.name,
                                        "description": a.description,
                                        "required": a.required,
                                    }
                                )
                                for a in p.arguments
                            ],
                        )
                    )
            except Exception as exc:
                logger.warning(f"⚠️  Could not convert prompt '{p.name}': {exc}")

        return prompt_objs

    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict | None = None) -> str:
        """Render a prompt by *name* using the internal registry."""
        try:
            rendered = render_prompt(name, arguments or {})
            return rendered
        except KeyError:
            raise ValueError(f"Prompt '{name}' not found.")

    @server.list_resources()
    async def handle_list_resources() -> list[types.Resource]:
        """List resources from the registry."""
        res_objs = []
        for meta in _list_resources():
            res_objs.append(
                types.Resource(
                    uri=meta.uri,
                    name=meta.name,
                    description=meta.description,
                    type=meta.type,
                )
            )
        return res_objs

    @server.read_resource()
    async def handle_read_resource(uri: str, **kwargs) -> str:
        """Delegate to resource registry adapters."""
        logger.info(f"📖 [MCP SERVER] Resource requested: {uri}")
        try:
            mime, content = await _read_resource(uri, **kwargs)
            return content
        except Exception as e:
            logger.error(f"💥 [MCP SERVER] Error reading resource {uri}: {e}")
            raise

    return server


async def connect_to_downstreams(stack: AsyncExitStack) -> tuple[list[types.Tool], dict[str, ClientSession], dict[str, ClientSession]]:
    """Connect to downstream servers defined in mcp_server_config.json."""
    config_path = Path("mcp_server_config.json")
    if not config_path.exists():
        logger.info(f"No {config_path} found. running in standalone mode.")
        return [], {}, {}

    config = json.loads(config_path.read_text())
    mcp_servers = config.get("mcpServers", {})
    
    tools_acc = []
    sessions = {}
    tool_map = {} # name -> session

    for srv_name, srv_conf in mcp_servers.items():
        url = srv_conf.get("url")
        if not url:
            continue
            
        try:
            logger.info(f"🔗 [GATEWAY] Connecting to downstream '{srv_name}' at {url}")
            # Use the stack to enter valid contexts that persist
            sse_ctx = sse_client(url=url)
            read, write = await stack.enter_async_context(sse_ctx)
            
            session_ctx = ClientSession(read, write)
            session = await stack.enter_async_context(session_ctx)
            
            await session.initialize()
            
            # List tools
            res = await session.list_tools()
            logger.info(f"✅ [GATEWAY] Connected to '{srv_name}'. Found {len(res.tools)} tools.")
            
            for t in res.tools:
                tools_acc.append(t)
                tool_map[t.name] = session
            
            sessions[srv_name] = session
            
        except Exception as e:
            logger.error(f"❌ [GATEWAY] Failed to connect to '{srv_name}': {e}")
            
    return tools_acc, tool_map, sessions

async def run_server():
    """Run the MCP server via HTTP/SSE."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount
    from starlette.responses import Response
    from starlette.requests import Request
    from mcp.server.sse import SseServerTransport

    setup_logger()
    logger.info("🚀 [MCP SERVER] Starting APE MCP Server via HTTP/SSE...")

    # Initialize Vector Memory
    await get_vector_memory()

    async with AsyncExitStack() as stack:
        # 1. Connect to Downstreams (Gateway)
        ds_tools, ds_tool_map, ds_sessions = await connect_to_downstreams(stack)

        # 2. Get the existing, fully configured MCP Server instance
        server = create_mcp_server(downstream_tools=ds_tools, downstream_sessions=ds_tool_map)

        # 3. Create an SSE transport
        sse_transport = SseServerTransport("/mcp/messages")

        async def handle_sse_connection(scope, receive, send):
            async with sse_transport.connect_sse(scope, receive, send) as streams:
                await server.run(
                    streams[0],
                    streams[1],
                    server.create_initialization_options(),
                )
            return Response(status_code=204)

        async def sse_endpoint(request: Request):
            return await handle_sse_connection(request.scope, request.receive, request._send)

        app = Starlette(routes=[
            Route("/mcp/sse", endpoint=sse_endpoint, methods=["GET"]),
            Mount("/mcp/messages", app=sse_transport.handle_post_message),
        ])

        config = uvicorn.Config(app, host="0.0.0.0", port=settings.PORT)
        uv_server = uvicorn.Server(config)
        logger.info(f"📡 [MCP SERVER] Starting HTTP server on port {settings.PORT}...")
        
        # Run until shutdown
        await uv_server.serve()



if __name__ == "__main__":
    asyncio.run(run_server())