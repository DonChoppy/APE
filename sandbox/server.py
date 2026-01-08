
import asyncio
import os
import subprocess
import sys
import json
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import Response
from starlette.requests import Request

import mcp.types as types
from mcp.server import Server
from mcp.server.sse import SseServerTransport

# Initialize the Sandbox Server
server = Server("ape-sandbox")

WORK_DIR = os.environ.get("ALLOWED_ROOT", "/app/workdir")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List tools available in the sandbox."""
    return [
        types.Tool(
            name="execute_shell",
            description="Execute a shell command in the sandbox. Use this for OS-level tasks. CWD is /app/workdir.",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run (e.g. 'ls -la', 'git clone ...')"},
                },
                "required": ["command"],
            },
        ),
        types.Tool(
            name="execute_python",
            description="Execute Python code in the sandbox. Returns stdout/stderr.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The Python code to execute."},
                },
                "required": ["code"],
            },
        ),
        types.Tool(
            name="read_file",
            description="Read a file from the sandbox filesystem.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file (e.g. 'output.txt')."},
                },
                "required": ["path"],
            },
        ),
        types.Tool(
            name="write_file",
            description="Write content to a file in the sandbox.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."},
                    "content": {"type": "string", "description": "Text content to write."},
                },
                "required": ["path", "content"],
            },
        ),
    ]

def _resolve_path(path: str) -> Path:
    """Securely resolve path to ensure it stays within WORK_DIR."""
    # Simple jail to prevent path traversal
    base = Path(WORK_DIR).resolve()
    target = (base / path).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError(f"Access denied: Path '{path}' is outside the sandbox root.")
    return target

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if not arguments:
        arguments = {}
    
    try:
        if name == "execute_shell":
            command = arguments.get("command", "")
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=WORK_DIR,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            result = {
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
                "returncode": process.returncode,
            }
            return [types.TextContent(type="text", text=json.dumps(result))]

        elif name == "execute_python":
            code = arguments.get("code", "")
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-c", code,
                cwd=WORK_DIR,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            result = {
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
                "returncode": process.returncode,
            }
            return [types.TextContent(type="text", text=json.dumps(result))]

        elif name == "read_file":
            path_str = arguments.get("path", "")
            target = _resolve_path(path_str)
            if not target.exists():
                return [types.TextContent(type="text", text=f"Error: File '{path_str}' does not exist.")]
            content = target.read_text(encoding="utf-8")
            return [types.TextContent(type="text", text=content)]

        elif name == "write_file":
            path_str = arguments.get("path", "")
            content = arguments.get("content", "")
            target = _resolve_path(path_str)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return [types.TextContent(type="text", text=f"Successfully wrote to '{path_str}'.")]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

async def run():
    # SSE Transport logic
    sse = SseServerTransport("/mcp/messages")

    async def handle_sse(scope, receive, send):
        async with sse.connect_sse(scope, receive, send) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())
        return Response(status_code=204)

    async def sse_endpoint(req: Request):
        return await handle_sse(req.scope, req.receive, req._send)

    app = Starlette(routes=[
        Route("/mcp/sse", endpoint=sse_endpoint, methods=["GET"]),
        Mount("/mcp/messages", app=sse.handle_post_message),
    ])

    config = uvicorn.Config(app, host="0.0.0.0", port=8080)
    server_uv = uvicorn.Server(config)
    await server_uv.serve()

if __name__ == "__main__":
    asyncio.run(run())
