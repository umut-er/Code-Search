import os
import sys
import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, List, Dict, Optional

os.environ["ANYIO_BACKEND"] = "asyncio"

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver 
from pydantic import create_model, Field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

MCP_SERVERS = [
    StdioServerParameters(
        command="npx",
        args=["-y", "@zilliz/claude-context-mcp@latest"],
        env=os.environ.copy()
    )
]

class SimpleMCPClient:
    def __init__(self, session: ClientSession):
        self.session = session

    async def mcp_tool_wrapper(self, name: str, **kwargs) -> str:
        print(f"\n🔵 [AGENT CALL] {name}")
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        print(f"   Args: {json.dumps(clean_kwargs, indent=2)}")
        
        try:
            result = await self.session.call_tool(name, arguments=clean_kwargs)
            
            final_text = ""
            if result.content:
                text_content = []
                for block in result.content:
                    if block.type == "text":
                        text_content.append(block.text)
                final_text = "\n".join(text_content)
            else:
                final_text = "<No Text Content Returned>"

            print(f"🟢 [MCP RESPONSE] ({len(final_text)} chars)")
            return final_text
            
        except Exception as e:
            print(f"🔴 [MCP ERROR]: {e}")
            return f"Error: {e}"

    def convert_to_langchain_tool(self, mcp_tool_def: Any) -> StructuredTool:
        name = mcp_tool_def.name
        description = mcp_tool_def.description or f"MCP Tool: {name}"
        schema = mcp_tool_def.inputSchema
        
        required_fields = schema.get("required", [])
        fields = {}
        
        if "properties" in schema:
            for field_name, field_info in schema["properties"].items():
                t = field_info.get("type")
                if t == "integer": field_type = int
                elif t == "number": field_type = float
                elif t == "boolean": field_type = bool
                elif t == "array": field_type = List[Any]
                elif t == "object": field_type = Dict[str, Any]
                else: field_type = str
                
                if field_name in required_fields:
                    fields[field_name] = (field_type, Field(..., description=field_info.get("description", "")))
                else:
                    fields[field_name] = (Optional[field_type], Field(default=None, description=field_info.get("description", "")))
        
        ArgsModel = create_model(f"{name}Args", **fields)

        async def tool_func(**kwargs):
            return await self.mcp_tool_wrapper(name, **kwargs)

        return StructuredTool.from_function(
            func=None,
            coroutine=tool_func,
            name=name,
            description=description,
            args_schema=ArgsModel
        )

async def run_agent_loop():
    print("🚀 Starting Agent (MCP Only)...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found")
        return

    async with AsyncExitStack() as stack:
        langchain_tools = []
        
        # Load Semantic MCP Tools (Zilliz)
        print("🔌 Connecting to MCP Servers...")
        for params in MCP_SERVERS:
            try:
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                
                client_wrapper = SimpleMCPClient(session)
                mcp_tools_list = await session.list_tools()
                
                for tool_def in mcp_tools_list.tools:
                    lc_tool = client_wrapper.convert_to_langchain_tool(tool_def)
                    langchain_tools.append(lc_tool)
                    print(f"✅ Loaded MCP Tool: {lc_tool.name}")
            except Exception as e:
                print(f"❌ Failed to connect to MCP: {e}")

        # Setup Agent
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        memory = MemorySaver() 

        # Updated system prompt to reflect the removal of local tools
        system_prompt = (
            f"You are an expert Coding Agent working in: {os.getcwd()}\n"
            "You are connected to an MCP (Model Context Protocol) server which provides your tools.\n"
            "You need to index a codebase before running semantic search.\n"
            "Use the available tools to semantic search, analyze, and understand the codebase.\n"
            "If you are using the semantic search tool, set the limit parameter to 5."
        )

        graph = create_agent(
            llm, 
            langchain_tools, 
            system_prompt=system_prompt, 
            checkpointer=memory
        )

        print("\n🤖 Agent Ready. Type 'exit' to quit.")
        print("=" * 60)

        config = {"configurable": {"thread_id": "main_session"}}
        
        total_input_tokens = 0
        total_output_tokens = 0

        while True:
            try:
                user_input = input("\nUser: ").strip()
                if user_input.lower() in ["exit", "quit"]: break
                if not user_input: continue
                
                print("\n(Processing...)\n")
                async for event in graph.astream_events(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                    version="v2"
                ):
                    kind = event["event"]
                    
                    if kind == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        if hasattr(chunk, "content") and chunk.content:
                            print(chunk.content, end="", flush=True)

                    elif kind == "on_chat_model_end":
                        output = event["data"]["output"]
                        if hasattr(output, "usage_metadata") and output.usage_metadata:
                            usage = output.usage_metadata
                            in_tokens = usage.get("input_tokens", 0)
                            out_tokens = usage.get("output_tokens", 0)
                            
                            total_input_tokens += in_tokens
                            total_output_tokens += out_tokens
                            
                            print(f"\n\n📊 [Usage] Input: {in_tokens} | Output: {out_tokens}")

                    elif kind == "on_tool_start":
                        print(f"\n\n🛠️  [TOOL] Calling {event['name']}...")
                        
                print("\n" + "-" * 60)
                print(f"💰 Session Total: {total_input_tokens} In | {total_output_tokens} Out")
                print("-" * 60)
                
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    asyncio.run(run_agent_loop())
