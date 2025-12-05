import os
import json
import time
import datetime
import asyncio
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

# Load environment variables from .env file
load_dotenv()

# Import existing local tools to make scanning generic
from src.local_tools import LocalTools

# Define the output schema for the route discovery
class RouteInfo(Dict):
    path: str
    component: str

# --- LOGGER SETUP ---
class RouteDiscoveryLogger:
    def __init__(self, base_dir="./route_discovery_logs"):
        self.base_dir = base_dir
        self.session_id = f"route_discovery_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join(self.base_dir, self.session_id)
        os.makedirs(self.log_dir, exist_ok=True)
        self.trace_bucket = []
        self.token_stats = {"input": 0, "output": 0, "total": 0}
        self.model_signature = "unknown_model"
        print(f"📁 Logging session to: {self.log_dir}")

    def add_event(self, event_type: str, data: dict):
        """Add an event to the trace bucket."""
        event = {
            "type": event_type,
            "timestamp": time.time(),
            **data
        }
        self.trace_bucket.append(event)

    def save_log(self, cwd: str, success: bool, routes_found: Optional[List] = None):
        """Save the complete log to a JSON file."""
        log_data = {
            "meta": {
                "session_id": self.session_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "cwd": cwd,
                "success": success,
                "routes_found_count": len(routes_found) if routes_found else 0,
            },
            "llm_signature": {
                "model": self.model_signature,
                "token_usage": self.token_stats
            },
            "trace": self.trace_bucket,
            "routes": routes_found if routes_found else []
        }
        
        filepath = os.path.join(self.log_dir, "complete_log.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, default=str)
        
        print(f"💾 Complete log saved to: {filepath}")
        return filepath

SYSTEM_PROMPT = """
You are a Senior Frontend Software Engineer and Route Discovery Expert. 
Your GOAL: Find all client-side routes in this frontend codebase and extract their URL paths and associated Component filenames.
Your Current Working Directory is: {cwd}

### INSTRUCTIONS
1. **Explore**: Use `find_file` or `grep_text` to look for router configurations.
   - Keywords to search: "react-router", "createBrowserRouter", "<Route", "path=", "routes.ts", "App.tsx".
2. **Read**: Use `read_file` to inspect the content of the router files.
3. **Extract**: Identify the mapping between URL paths (e.g., "/login") and Component files (e.g., "Login.tsx").
   - If the component is lazy loaded or imported, try to resolve the filename.
4. **Finish**: When you are confident you have found the main routes, output the JSON list using the `save_routes` tool.

### OUTPUT FORMAT
The `save_routes` tool expects a JSON string:
[
  {{"path": "/login", "component": "src/pages/Login.tsx"}},
  {{"path": "/dashboard", "component": "src/features/Dashboard.tsx"}}
]
"""

@tool("save_routes")
def save_routes(routes_json: str) -> str:
    """
    Call this tool FINAL tool when you have extracted the routes.
    Args:
        routes_json: A JSON string list of objects with 'path' and 'component'.
    """
    try:
        data = json.loads(routes_json)
        
        with open("routes.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        return "SUCCESS: Routes saved to routes.json. You can stop now."
    except Exception as e:
        return f"ERROR: Invalid JSON format. {e}"

async def _run_route_discovery_async(enable_logging: bool = False):
    """
    Async helper function that runs the agent with event streaming for logging.
    """
    logger = RouteDiscoveryLogger() if enable_logging else None
    
    tools = [
        LocalTools.find_file,
        LocalTools.grep_text,
        LocalTools.read_file,
        save_routes
    ]

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    graph = create_react_agent(llm, tools)
    
    cwd = os.path.join(os.getcwd(), "bilkent-tanitim", "frontend")
    print(f"📂 Analyzing codebase at: {cwd}")
    
    if logger:
        logger.add_event("session_start", {"cwd": cwd})
    
    formatted_prompt = SYSTEM_PROMPT.format(cwd=cwd)
    user_message = "Find all routes in this project and save them."
    
    inputs = {
        "messages": [
            SystemMessage(content=formatted_prompt),
            ("user", user_message)
        ]
    }
    
    if logger:
        logger.add_event("prompt_sent", {
            "system_prompt": formatted_prompt[:500] + "..." if len(formatted_prompt) > 500 else formatted_prompt,
            "user_message": user_message
        })
    
    config = {"recursion_limit": 15}
    
    try:
        if enable_logging and logger:
            # Use async streaming for detailed logging
            async for event in graph.astream_events(inputs, config=config, version="v2"):
                kind = event["event"]
                data = event.get("data", {})
                name = event.get("name", "unknown")
                
                # Extract token usage and model info
                if kind == "on_chat_model_end":
                    output = data.get("output")
                    if output:
                        usage = getattr(output, "usage_metadata", {}) or {}
                        
                        if not usage and hasattr(output, "response_metadata"):
                            meta = output.response_metadata
                            usage = meta.get("token_usage") or meta.get("usage") or {}
                        
                        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
                        total = usage.get("total_tokens") or (input_tokens + output_tokens)
                        
                        logger.token_stats["input"] += input_tokens
                        logger.token_stats["output"] += output_tokens
                        logger.token_stats["total"] += total
                        
                        if logger.model_signature == "unknown_model":
                            logger.model_signature = (
                                getattr(output, "name", None) or 
                                (hasattr(output, "response_metadata") and output.response_metadata.get("model_name")) or 
                                (hasattr(output, "response_metadata") and output.response_metadata.get("model")) or 
                                "unknown_model"
                            )
                        
                        content = getattr(output, "content", "") or str(output)
                        logger.add_event("llm_response", {
                            "content": str(content)[:1000] + "..." if len(str(content)) > 1000 else str(content),
                            "token_usage": {
                                "input": input_tokens,
                                "output": output_tokens,
                                "total": total
                            }
                        })
                
                # Log tool calls
                elif kind == "on_tool_start":
                    tool_input = data.get("input", {})
                    
                    logger.add_event("tool_call", {
                        "tool": name,
                        "input": str(tool_input)[:500] if len(str(tool_input)) > 500 else str(tool_input)
                    })
                    
                    print(f"      🔧 Tool: {name}")
                
                elif kind == "on_tool_end":
                    tool_output = data.get("output", "")
                    
                    output_str = str(tool_output)
                    truncated_output = output_str[:1000] + "..." if len(output_str) > 1000 else output_str
                    
                    logger.add_event("tool_result", {
                        "tool": name,
                        "output": truncated_output
                    })
        else:
            # Simple invocation without logging
            result = graph.invoke(inputs, config=config)
            
        # Check if routes.json was created
        routes_found = None
        if os.path.exists("routes.json"):
            with open("routes.json", "r", encoding="utf-8") as f:
                routes_found = json.load(f)
        
        success = routes_found is not None and len(routes_found) > 0
        
        if logger:
            logger.add_event("session_end", {"success": success, "routes_count": len(routes_found) if routes_found else 0})
            logger.save_log(cwd, success, routes_found)
        
        return routes_found
        
    except Exception as e:
        error_msg = str(e)
        print(f"      ❌ Error: {error_msg}")
        
        if logger:
            logger.add_event("error", {"message": error_msg, "type": type(e).__name__})
            logger.save_log(cwd, False, None)
        
        raise

def run_route_discovery(enable_logging: bool = False):
    """
    Runs a specialized agent to discover routes generically using codebase tools.
    
    Args:
        enable_logging: If True, enables detailed logging to file.
    """
    print("🚀 Spawning Route Discovery Agent...")
    
    if enable_logging:
        # Run async version for detailed logging
        routes = asyncio.run(_run_route_discovery_async(enable_logging=True))
    else:
        # Simple synchronous version
        tools = [
            LocalTools.find_file,
            LocalTools.grep_text,
            LocalTools.read_file,
            save_routes
        ]

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        graph = create_react_agent(llm, tools)
        
        cwd = os.path.join(os.getcwd(), "bilkent-tanitim", "frontend")
        print(f"📂 Analyzing codebase at: {cwd}")
        
        formatted_prompt = SYSTEM_PROMPT.format(cwd=cwd)
        inputs = {
            "messages": [
                SystemMessage(content=formatted_prompt),
                ("user", "Find all routes in this project and save them.")
            ]
        }
        
        graph.invoke(inputs, config={"recursion_limit": 15})
    
    print("✅ Route Discovery Process Completed.")