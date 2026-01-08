#!/usr/bin/env python3
"""
APE CLI Chat Interface

A command-line interface that uses the official MCP Python SDK to connect to the APE MCP server.
This demonstrates proper MCP client/server communication.
"""

# ---------------------------------------------------------------------------
# Load .env before any other application code so that pydantic-settings
# picks up the correct environment variables.
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------

import asyncio
import sys
import uuid
import json
import os
import pprint
from typing import Optional, Dict, Any, List
from datetime import datetime
import importlib

import torch
from loguru import logger
import ollama
from ape.utils import setup_logger, count_tokens

from ape.mcp.session_manager import get_session_manager
from ape.cli.context_manager import ContextManager
from ape.cli.mcp_client import MCPClient
from ape.cli.chat_agent import ChatAgent
from ape.settings import settings
from ape.db_pool import close_all_pools

# Better CLI input handling (arrow keys, history)
try:
    from prompt_toolkit import PromptSession
except ImportError:  # fallback if library not installed
    PromptSession = None

class APEChatCLI:
    """Command-line interface for APE chat functionality using MCP."""
    
    def __init__(self):
        # ------------------------------------------------------------------
        # Bootstrapping: configure log sinks once per process
        # ------------------------------------------------------------------
        setup_logger()

        self.session_id = str(uuid.uuid4())
        self.session_manager = get_session_manager()
        # Wrapper that manages the underlying MCP stdio connection
        self.mcp_client = MCPClient()
        # self.mcp_session removal: we now rely on mcp_client wrapper exclusively
        self.context_manager = ContextManager(self.session_id)
        self.chat_agent = ChatAgent(self.session_id, self.mcp_client, self.context_manager)
        logger.info(f"Started new chat session: {self.session_id}")
        
        # Initialise prompt session for nicer input UX if available
        self.prompt: Optional[PromptSession] = None
        if PromptSession is not None:
            self.prompt = PromptSession()
    
    def print_banner(self):
        """Print the APE CLI banner."""
        print("\n" + "="*60)
        print("🤖 APE (Agentic Protocol Executor) - CLI Chat")
        print("="*60)
        print("Session ID:", self.session_id[:8] + "...")
        print("\nCommands:")
        print("  /help     - Show this help")
        print("  /history  - Show conversation history")
        print("  /session  - Show session info")
        print("  /tools    - List available MCP tools")
        print("  /context  - Show current session context")
        print("  /clear    - Clear screen")
        print("  /reset    - Clear session context")
        print("  /quit     - Exit chat")
        print("  /exit     - Exit chat")
        print("  /q        - Exit chat")
        print("  /errors   - Show recent tool errors")
        print("  /memory   - Show WindowMemory summary & stats")
        print("\n🧠 Intelligence: Connected to MCP server with tools:")
        print("  • Database tools for conversation management")
        print("  • Search tools for finding content")
        print("  • History tools for context retrieval")
        print("="*60 + "\n")
    
    async def connect_to_mcp(self):
        """Connect to the MCP server through the reusable wrapper."""
        return await self.mcp_client.connect()
    
    async def disconnect_from_mcp(self):
        """Disconnect via the reusable wrapper."""
        await self.mcp_client.disconnect()
    
    async def list_tools(self):
        """List available MCP tools."""
        try:
            if not self.mcp_client.is_connected:
                print("❌ Not connected to MCP server")
                return
            
            logger.info("🔧 [MCP CLIENT] Requesting tool list from MCP server")
            tools_result = await self.mcp_client.list_tools()
            logger.info(f"✅ [MCP CLIENT] Received {len(tools_result.tools)} tools from server")
            
            print(f"\n🔧 Available MCP Tools ({len(tools_result.tools)}):")
            print("-" * 50)
            
            for tool in tools_result.tools:
                print(f"• {tool.name}: {tool.description}")
                
            print("-" * 50)
            
        except Exception as e:
            logger.error(f"❌ [MCP CLIENT] Error listing tools: {e}")
            print(f"❌ Error listing tools: {e}")
    
    async def show_history(self, limit: int = 10):
        """Show conversation history using MCP."""
        try:
            if not self.mcp_client.is_connected:
                print("❌ Not connected to MCP server")
                return
            
            logger.info(f"🔧 [MCP CLIENT] Calling get_conversation_history via MCP (session: {self.session_id}, limit: {limit})")
            result = await self.mcp_client.call_tool(
                "get_conversation_history", 
                {"session_id": self.session_id, "limit": limit}
            )
            
            if result.content:
                history_text = result.content[0].text
                logger.info(f"✅ [MCP CLIENT] History received, {len(history_text)} chars")
                
                if "No conversation history found" in history_text:
                    print("📭 No conversation history yet.")
                    return
                
                try:
                    logger.debug(f"Raw history_text: {history_text}")
                    envelope = json.loads(history_text)
                    logger.debug(f"Decoded envelope: {envelope}")
                    payload_str = envelope.get('payload')
                    if not payload_str:
                        raise ValueError("Missing 'payload' in tool response")
                    logger.debug(f"Payload string: {payload_str}")

                    tool_result = json.loads(payload_str)
                    logger.debug(f"Decoded tool_result: {tool_result}")
                    history = tool_result.get('result')
                    logger.debug(f"Final history object: {history}")
                    logger.debug(f"Type of history object: {type(history)}")

                    if not isinstance(history, list):
                         # If history is not a list, it might be an error message string
                         print(f"Received from tool: {history}")
                         return

                    print(f"\n📚 Last {len(history)} messages:")
                    print("-" * 50)
                    
                    for msg in history:
                        logger.debug(f"Processing message: {msg}")
                        logger.debug(f"Type of message: {type(msg)}")
                        role_icon = "👤" if msg.get("role") == "user" else "🤖"
                        timestamp = msg.get("timestamp", "unknown")[:19]
                        content = msg.get("content", "")
                        content_preview = content[:100] + "..." if len(content) > 100 else content
                        print(f"{role_icon} [{timestamp}] {content_preview}")
                    
                    print("-" * 50)
                    
                except (json.JSONDecodeError, TypeError, AttributeError, ValueError) as e:
                    logger.error(f"Error parsing history response: {e}\nRaw response: {history_text}")
                    print("📚 Recent conversation history (raw):")
                    print(history_text)
            else:
                print("📭 No conversation history yet.")
            
        except Exception as e:
            logger.error(f"❌ [MCP CLIENT] Error retrieving history: {e}")
            print(f"❌ Error retrieving history: {e}")
    
    async def show_session_info(self):
        """Show current session information."""
        try:
            sessions = await self.session_manager.a_get_all_sessions()
            current_session = next((s for s in sessions if s["session_id"] == self.session_id), None)
            
            print(f"\n🔍 Session Information:")
            print(f"  Session ID: {self.session_id}")
            
            if current_session:
                print(f"  Messages: {current_session['message_count']}")
                print(f"  First Message: {current_session['first_message']}")
                print(f"  Last Message: {current_session['last_message']}")
            else:
                print("  Status: New session (no messages yet)")
                
            print(f"  Total Sessions: {len(sessions)}")
            print(f"  MCP Connected: {'✅' if self.mcp_client.is_connected else '❌'}")
            
        except Exception as e:
            print(f"❌ Error getting session info: {e}")
    
    async def discover_capabilities(self) -> dict:
        """Discover available MCP capabilities dynamically."""
        capabilities = {
            "tools": [],
            "prompts": [],
            "resources": []
        }
        
        try:
            # Discover available tools
            tools_result = await self.mcp_client.list_tools()
            capabilities["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
                for tool in tools_result.tools
            ]
            
            # Discover available prompts (new SDKs) – with graceful fallback
            try:
                prompts_result = await self.mcp_client.list_prompts()
                prompt_items = getattr(prompts_result, "prompts", prompts_result)
                capabilities["prompts"] = [
                    {
                        "name": p.name,
                        "description": p.description,
                        "arguments": [getattr(arg, "dict", lambda: arg)() for arg in getattr(p, "arguments", [])]
                    }
                    for p in prompt_items
                ]
            except Exception as exc:
                logger.debug(f"list_prompts failed ({exc}); using local registry fallback.")
                try:
                    from ape.prompts import list_prompts as _local_list
                    capabilities["prompts"] = [
                        {
                            "name": prm.name,
                            "description": prm.description,
                            "arguments": [arg.dict() for arg in prm.arguments]
                        }
                        for prm in _local_list()
                    ]
                except Exception as loc_exc:
                    logger.debug(f"Local prompt registry fallback failed: {loc_exc}")
            
            # Discover available resources
            resources_result = await self.mcp_client.list_resources()
            capabilities["resources"] = [
                {
                    "name": resource.name,
                    "uri": resource.uri,
                    "description": resource.description,
                    "type": resource.type
                }
                for resource in resources_result.resources
            ]
            
            logger.info(f"🔍 [MCP CLIENT] Discovered capabilities: {json.dumps(capabilities, indent=2)}")
            return capabilities
            
        except Exception as e:
            logger.error(f"❌ [MCP CLIENT] Error discovering capabilities: {e}")
            return capabilities

    async def create_dynamic_system_prompt(self, capabilities: dict) -> str:
        """Create system prompt using the shared Prompt Registry."""

        from ape.prompts import render_prompt  # lazy import to avoid cycles

        def _fmt(items: list[dict], label: str = "") -> str:
            """Render items with optional argument hints."""
            if not items:
                return f"No {label}available".strip()

            def _args_hint(itm: dict) -> str:
                if "parameters" in itm and isinstance(itm["parameters"], dict):
                    props = itm["parameters"].get("properties", {})
                    if props:
                        return ", args: " + ", ".join(props.keys())
                if "arguments" in itm and itm["arguments"]:
                    return ", args: " + ", ".join(arg.get("name", "?") for arg in itm["arguments"])
                return ""

            if label:
                label = " " + label

            lines: list[str] = []
            for itm in items:
                lines.append(f"• **{itm['name']}**{label}{_args_hint(itm)}: {itm['description']}")
            return "\n".join(lines)

        tools_section = _fmt(capabilities["tools"])
        prompts_section = _fmt(capabilities["prompts"])
        resources_section = (
            "\n".join(
                [
                    f"• **{res['name']}** (`{res['uri']}`) [{res['type']}]: {res['description']}"
                    for res in capabilities["resources"]
                ]
            )
            if capabilities["resources"]
            else "No resources available"
        )

        base_prompt = render_prompt(
            "system",
            {
                "agent_name": "APE",
                "current_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tools_section": tools_section,
                "prompts_section": prompts_section,
                "resources_section": resources_section,
            },
        )

        return base_prompt

    async def chat_with_llm(self, message: str, conversation: list) -> str:
        """Send a message to the LLM with enhanced autonomous capabilities."""
        try:
            # Discover available capabilities
            capabilities = await self.discover_capabilities()
            
            # Create dynamic system prompt
            system_prompt = await self.create_dynamic_system_prompt(capabilities)
            
            # Add session context if available
            context_summary = self.context_manager.get_context_summary()
            if context_summary.strip() != "CURRENT SESSION CONTEXT:":
                system_prompt += f"\n\nCURRENT CONTEXT:\n{context_summary}"
            
            # Increased iteration limits for autonomous operation
            max_iterations = settings.MAX_TOOLS_ITERATIONS
            current_iteration = 0
            cumulative_response = ""
            
            # Initialize conversation with user's message
            execution_conversation = [
                {"role": "system", "content": system_prompt},
                *conversation,
                {"role": "user", "content": message}
            ]
            
            ollama = importlib.import_module("ollama")
            client = ollama.AsyncClient(host=str(settings.OLLAMA_BASE_URL))
            
            while current_iteration < max_iterations:
                current_chunk = ""
                has_tool_calls = False
                
                stream = await client.chat(
                    model=settings.LLM_MODEL,
                    messages=execution_conversation,
                    tools=capabilities["tools"],
                    options={"temperature": settings.TEMPERATURE,
                             "top_p": settings.TOP_P,
                             "top_k": settings.TOP_K},
                    stream=True
                )
                
                turn_messages = []
                
                async for chunk in stream:
                    if 'message' in chunk:
                        if chunk['message'].get('content', ''):
                            content = chunk['message']['content']
                            # Print ALL content including thinking - don't filter anything
                            print(content, end="", flush=True)
                            current_chunk += content
                        
                        if chunk['message'].get('tool_calls'):
                            has_tool_calls = True
                            current_iteration += 1
                            
                            # Save the assistant's complete response (including thinking)
                            if current_chunk:
                                turn_messages.append({
                                    "role": "assistant",
                                    "content": current_chunk,
                                    "timestamp": ""
                                })
                                cumulative_response += current_chunk + "\n"
                            
                            # Handle tools with enhanced context
                            tool_results = await self.handle_tool_calls(chunk['message']['tool_calls'])
                            print("\n" + tool_results)
                            
                            turn_messages.append({
                                "role": "tool", 
                                "content": tool_results,
                                "timestamp": ""
                            })
                            
                            # Add tool response to conversation for LLM to continue reasoning
                            execution_conversation.extend([
                                {"role": "assistant", "content": current_chunk},
                                {"role": "tool", "content": tool_results}
                            ])
                            
                            # Let the LLM continue naturally - no forced prompts
                            # The LLM will decide what to do next based on the tool results
                            
                            current_chunk = ""
                            break
                
                if not has_tool_calls:
                    # Final response - save it
                    if current_chunk:
                        cumulative_response += current_chunk
                        turn_messages.append({
                            "role": "assistant",
                            "content": current_chunk,
                            "timestamp": ""
                        })
                    break
                
                if current_iteration >= max_iterations:
                    warning = "\n\n[Note: Reached maximum iteration limit. Task may require continuation.]"
                    print(warning)
                    cumulative_response += warning
                    turn_messages.append({
                        "role": "assistant",
                        "content": warning,
                        "timestamp": ""
                    })
                    break
            
            # Save complete conversation including thinking process
            # Get existing history and append current turn
            existing_history = await self.session_manager.a_get_history(self.session_id)
            current_turn = [
                {"role": "user", "content": message, "timestamp": ""},
                *turn_messages
            ]
            all_messages = existing_history + current_turn
            await self.session_manager.a_save_messages(self.session_id, all_messages)
            
            print()
            return cumulative_response
                
        except Exception as e:
            logger.error(f"💥 [MCP CLIENT] Error in chat_with_llm: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    async def get_ollama_tools(self) -> list:
        """Get tools in Ollama format."""
        if not self.mcp_client.is_connected:
            return []
        
        try:
            tools_result = await self.mcp_client.list_tools()
            ollama_tools = []
            
            for tool in tools_result.tools:
                ollama_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                }
                ollama_tools.append(ollama_tool)
            
            return ollama_tools
            
        except Exception as e:
            logger.error(f"Error getting Ollama tools: {e}")
            return []
    
    async def handle_tool_calls(self, tool_calls: list) -> str:
        """Handle tool calls from the LLM using MCP protocol with streaming support."""
        if not self.mcp_client.is_connected:
            return "Sorry, I'm not connected to the MCP server right now."
        
        try:
            results = []
            
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
                
                # Check if we can substitute extracted values for placeholders
                if isinstance(arguments, dict):
                    for key, value in arguments.items():
                        if isinstance(value, str):
                            # Replace common placeholders with actual values
                            if value == "retrieved_session_id" and "last_session_id" in self.context_manager.extracted_values:
                                arguments[key] = self.context_manager.extracted_values["last_session_id"]
                                logger.info(f"🔄 [MCP CLIENT] Substituted {value} with {arguments[key]}")
                            elif value == "last_count" and "last_count" in self.context_manager.extracted_values:
                                arguments[key] = self.context_manager.extracted_values["last_count"]
                                logger.info(f"🔄 [MCP CLIENT] Substituted {value} with {arguments[key]}")
                
                # Auto-substitute current session ID for interaction tools if not provided
                if function_name in ["get_last_N_user_interactions", "get_last_N_tool_interactions", "get_last_N_agent_interactions"]:
                    if "session_id" not in arguments or not arguments.get("session_id"):
                        arguments["session_id"] = self.context_manager.current_session_id
                        logger.info(f"🔄 [MCP CLIENT] Auto-substituted current session ID: {arguments['session_id']}")
                
                logger.info(f"🔧 [MCP CLIENT] Executing tool: {function_name} with args: {arguments}")
                
                # Call the MCP tool
                result = await self.mcp_client.call_tool(function_name, arguments)
                
                if result.content:
                    tool_result = result.content[0].text
                    results.append({
                        "tool": function_name,
                        "arguments": arguments,
                        "result": tool_result
                    })
                    
                    # Add to context manager for value extraction
                    self.context_manager.add_tool_result(function_name, arguments, tool_result)
                    
                    logger.info(f"✅ [MCP CLIENT] Tool {function_name} executed successfully")
                else:
                    results.append({
                        "tool": function_name,
                        "arguments": arguments,
                        "result": "No results returned"
                    })
            
            # Format results with validation and error detection
            formatted_response = "🔧 Tool Execution Results:\n\n"
            has_valid_data = False
            
            for idx, result in enumerate(results, 1):
                formatted_response += f"**Tool {idx}: {result['tool']}**\n"
                formatted_response += f"Arguments: {result['arguments']}\n"
                
                # Validate tool result and detect issues
                tool_result = result['result']
                is_error = any(indicator in tool_result.lower() for indicator in [
                    "error", "failed", "exception", "no results found", "rows affected: -1"
                ])
                
                if is_error:
                    formatted_response += f"⚠️ **ISSUE DETECTED**: {tool_result}\n"
                    formatted_response += "❗ This tool did not return valid data. Do not use invented data.\n\n"
                else:
                    try:
                        # Attempt to decode the JWT-like envelope
                        envelope = json.loads(tool_result)
                        payload_str = envelope.get('payload')
                        if not payload_str:
                            raise ValueError("Missing 'payload' in tool response")
                        
                        # The payload itself is a JSON string, decode it
                        final_data = json.loads(payload_str)
                        
                        # Now pretty-print the actual result object
                        pretty_result = pprint.pformat(final_data, indent=2)
                        formatted_response += f"Result:\n{pretty_result}\n\n"

                    except (json.JSONDecodeError, TypeError, ValueError):
                        # If anything fails, just print the raw result
                        formatted_response += f"Result: {tool_result}\n\n"
                    
                    has_valid_data = True
            
            # Add explicit warning if no valid data was obtained
            if not has_valid_data:
                formatted_response += """
🚨 **CRITICAL WARNING**: None of the tools returned valid data.
- DO NOT invent or assume any numbers, facts, or information
- Inform the user about the tool execution issues
- Ask for help debugging the problem or suggest alternative approaches
- NEVER present made-up data as if it came from the tools
"""
            
            # Add key extracted values that might be useful for next steps
            key_values = {}
            for key, value in self.context_manager.extracted_values.items():
                if key in ["last_session_id", "last_message_count", "total_messages", "total_sessions"]:
                    key_values[key] = value
            
            if key_values:
                formatted_response += "📊 Key Values Available:\n"
                for key, value in key_values.items():
                    formatted_response += f"- {key}: {value}\n"
                formatted_response += "\n"
            
            logger.info(f"✅ [MCP CLIENT] Successfully executed {len(results)} tools")
            return formatted_response
                
        except Exception as e:
            logger.error(f"❌ [MCP CLIENT] Error handling tool calls: {e}")
            return f"I encountered an error while using tools: {str(e)}"
    
    async def show_tools(self):
        """Show available MCP tools."""
        await self.list_tools()  # Reuse existing list_tools implementation
    
    def clear_context(self):
        """Clear the current session context."""
        self.context_manager.clear()
        print("🧹 Session context cleared.")
    
    def show_help(self):
        """Display help information."""
        print("""
🤖 APE (Agentic Protocol Executor) - Enhanced Autonomous Agent

Available commands:
  /help     - Show this help
  /history  - Show conversation history  
  /session  - Show session info
  /tools    - List available MCP tools
  /context  - Show current session context
  /clear    - Clear screen
  /reset    - Clear session context
  /quit     - Exit chat
  /exit     - Exit chat
  /q        - Exit chat
  /errors   - Show recent tool errors
  /memory   - Show WindowMemory summary & stats

🚀 Enhanced Autonomous Capabilities:
The agent can now handle complex multi-step tasks naturally by:
• Thinking through problems step-by-step
• Chaining multiple tool calls automatically
• Building upon previous results
• Completing comprehensive analysis tasks

Try complex requests like:
• \"Get database info, find a random session and analyze its conversation patterns\"
• \"Search for recent tool usage and compare it with user activity patterns\"
• \"Analyze the database structure and provide insights about conversation trends\"
• \"Find the most active sessions and summarize their content\"

The agent will use its natural reasoning to break down complex tasks!
""")

    async def show_context(self):
        """Show current session context."""
        context_summary = self.context_manager.get_context_summary()
        print("\n" + "="*60)
        print("📋 SESSION CONTEXT")
        print("="*60)
        print(context_summary)
        print("="*60)
    
    async def show_errors(self, limit: int = 20):
        """Display recent structured tool errors from the Error Bus."""
        try:
            sm = get_session_manager()
            errors = await sm.a_get_recent_errors(limit, session_id=self.session_id)

            if not errors:
                print("\n✅ No tool errors recorded.")
                return

            print(f"\n🚨 Last {len(errors)} tool error(s) for this session:")
            print("-" * 60)
            for err in errors:
                tool = err.get("tool")
                ts = str(err.get("timestamp"))[:19]
                msg = err.get("error", "")
                args_preview = json.dumps(err.get("arguments", {}))[:80]
                print(f"[{ts}] {tool}  args={args_preview}\n   → {msg}\n")
            print("-" * 60)

        except Exception as exc:
            logger.error(f"Error fetching tool errors: {exc}")
            print(f"❌ Error fetching tool errors: {exc}")
    
    async def show_memory(self):
        """Display WindowMemory stats and current summary."""
        try:
            mem = getattr(self.chat_agent, "memory", None)
            if mem is None:
                print("❌ Memory module not initialised.")
                return

            raw_msg_count = len(mem.messages)
            raw_tokens = sum(count_tokens(m["content"]) for m in mem.messages)
            summary_tokens = count_tokens(mem.summary)
            total = mem.tokens()

            print("\n🧠 WINDOW MEMORY STATUS")
            print("-" * 60)
            print(f"Raw messages kept : {raw_msg_count}")
            print(f"Raw tokens        : {raw_tokens}")
            print(f"Summary tokens    : {summary_tokens}")
            print(f"Total tokens      : {total}")
            print("-" * 60)
            print("CURRENT SUMMARY: \n")
            print(mem.latest_context())
            print("\n" + "-" * 60)
        except Exception as exc:
            logger.error(f"Error displaying memory: {exc}")
            print(f"❌ Error displaying memory: {exc}")
    
    async def run(self):
        """Main chat loop."""
        self.print_banner()
        
        # Connect to MCP server
        if not await self.connect_to_mcp():
            print("❌ Cannot continue without MCP server connection")
            return

        await self.chat_agent.initialize()
        agent_card = self.chat_agent.get_agent_card()
        logger.debug(f"Agent card:\n{pprint.pformat(agent_card)}")
        
        try:
            while True:
                try:
                    # Use prompt_toolkit for a better CLI experience if available
                    if self.prompt:
                        try:
                            user_input = await self.prompt.prompt_async("\nYou: ")
                        except EOFError:
                            # Exit gracefully on Ctrl+D
                            break
                    else:
                        # Fallback to a simpler, more robust input method.
                        print("\nYou: ", end="", flush=True)
                        loop = asyncio.get_running_loop()
                        user_input = (await loop.run_in_executor(
                            None, sys.stdin.readline
                        )).strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.startswith('/'):
                        if user_input == '/quit' or user_input == '/exit' or user_input == '/q':
                            break
                        elif user_input == '/help':
                            self.show_help()
                        elif user_input == '/history':
                            await self.show_history()
                        elif user_input == '/session':
                            await self.show_session_info()
                        elif user_input == '/tools':
                            await self.show_tools()
                        elif user_input == '/clear':
                            print("\033[2J\033[H")
                        elif user_input == '/context':
                            await self.show_context()
                        elif user_input == '/reset':
                            self.clear_context()
                        elif user_input == '/errors':
                            await self.show_errors()
                        elif user_input == '/memory':
                            await self.show_memory()
                        continue
                    
                    # Process regular message
                    print("🤖 APE: ", end="", flush=True)
                    
                    # Get conversation history for context
                    conversation = []
                    try:
                        history = await self.session_manager.a_get_history(self.session_id)
                        for msg in history:
                            conversation.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })
                    except Exception as e:
                        logger.warning(f"Could not retrieve history: {e}")
                    
                    response = await self.chat_agent.chat_with_llm(user_input, conversation)
                    
                    # Ensure the cursor moves to the next line before showing the prompt again
                    if not response.endswith("\n"):
                        print()
                    
                    # Save user + assistant turn in history
                    try:
                        existing_history = await self.session_manager.a_get_history(self.session_id)
                        turn = [
                            {"role": "user", "content": user_input, "timestamp": ""},
                            {"role": "assistant", "content": response, "timestamp": ""},
                        ]
                        await self.session_manager.a_save_messages(self.session_id, existing_history + turn)
                    except Exception as e:
                        logger.error(f"Error saving chat history: {e}")
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Chat interrupted. Use /quit or /exit or /q to exit gracefully.")
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    logger.error(f"Chat error: {e}")
        
        finally:
            await self.disconnect_from_mcp()
            # ------------------------------------------------------------------
            # Ensure all background resources (e.g. aiosqlite worker threads)
            # are cleaned up so the interpreter can exit without requiring
            # a manual Ctrl+C.
            # ------------------------------------------------------------------
            try:
                await close_all_pools()
                logger.info("🗃️  [DB POOL] All database connections closed")
            except Exception as exc:
                logger.warning(f"[DB POOL] Error during shutdown: {exc}")

            # Explicitly terminate the process to avoid lingering non-daemon threads
            # (e.g. those spawned by aiosqlite) keeping Python alive.
            sys.exit(0)


async def main():
    """Main entry point for the CLI chat."""
    
    print("🚀 Starting APE CLI Chat with MCP...")
    
    # Check if Ollama is available
    try:
        ollama = importlib.import_module("ollama")
        client = ollama.Client(host=str(settings.OLLAMA_BASE_URL))
        models = client.list()
        if not models.get('models'):
            print("⚠️  Warning: No Ollama models found. Make sure Ollama is running and has models installed.")
            print("   Try: ollama pull gemma3:4b")
    except Exception as e:
        print(f"⚠️  Warning: Could not connect to Ollama: {e}")
        print("   Make sure Ollama is running at http://localhost:11434")
    
    # Start the chat interface
    chat = APEChatCLI()
    await chat.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error in CLI: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)