import os
import asyncio
from dotenv import load_dotenv
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.messages import HumanMessage

# Import from src
from src.local_tools import LocalTools
from src.mcp_client import SimpleMCPClient
from src.agent import build_graph

load_dotenv()

MCP_SERVERS = [
    StdioServerParameters(
        command="npx",
        args=["-y", "@zilliz/claude-context-mcp@latest"],
        env=os.environ.copy()
    )
]

async def run_interactive():
    print("🚀 Starting Agent...")
    
    async with AsyncExitStack() as stack:
        tools = []
        
        # 1. Load Local Tools
        tools.extend([
            LocalTools.read_file,
            LocalTools.list_directory,
            LocalTools.search_code,
            LocalTools.get_code_symbols
        ])

        # 2. Load MCP Tools
        for params in MCP_SERVERS:
            try:
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                
                client_wrapper = SimpleMCPClient(session)
                mcp_tools = await session.list_tools()
                
                for t in mcp_tools.tools:
                    tools.append(client_wrapper.convert_to_langchain_tool(t))
                    print(f"✅ Loaded MCP: {t.name}")
            except Exception as e:
                print(f"❌ MCP Error: {e}")

        # 3. Build Graph
        graph = build_graph(tools)
        config = {"configurable": {"thread_id": "cli_session"}}

        print("\n🤖 Ready. Type 'exit' to quit.")
        
        while True:
            try:
                user_input = input("\nUser: ").strip()
                if user_input.lower() in ["exit", "quit"]: break
                if not user_input: continue
                
                async for event in graph.astream_events(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                    version="v2"
                ):
                    kind = event["event"]
                    if kind == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        if hasattr(chunk, "content"):
                            print(chunk.content, end="", flush=True)
                    elif kind == "on_tool_start":
                        print(f"\n🛠️  {event['name']}...", end="", flush=True)
                        
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    asyncio.run(run_interactive())
