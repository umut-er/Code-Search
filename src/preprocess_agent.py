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
from openai import RateLimitError, OpenAI
import re

# Load environment variables from .env file
load_dotenv()

# Import existing local tools to make scanning generic
from src.local_tools import LocalTools

# Global state to hold routes before final validation and saving
# This allows the agent to work with routes in memory without file path issues
_routes_state = {
    "routes": None,
    "file_path": None,
    "extracted_at": None
}

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
   - Keywords to search: "react-router", "createBrowserRouter", "<Route", "path=", "routes.ts", "router.tsx", "App.tsx".
2. **Find Router File**: Once you find the router file (usually named router.tsx, routes.ts, or App.tsx), 
   use `read_file` to read the COMPLETE file content (use end_line=-1 to read entire file).
3. **Find Constants**: After reading the router file, analyze it to determine if additional context is needed.
   - If you see constant references instead of actual paths (e.g., `ROUTES.LOGIN`, `CONSTANTS.HOME` instead of `"/login"`, `"/home"`),
     this means the router file imports constants from another file.
   - In this case, you MUST:
     a. Identify which constant object is being used (e.g., `ROUTES`, `CONSTANTS`, `ROUTE_PATHS`)
     b. Check the import statements in the router file to find where these constants are imported from
     c. Use `find_file` or `grep_text` to locate the constants definition file
     d. Read the COMPLETE constants file using `read_file` (use end_line=-1 to read entire file)
     e. Pass the full content of the constants file as `helpful_context` parameter to `extract_routes_from_file`
   - Example: If router file shows `path: ROUTES.LOGIN` and imports `ROUTES from "@/constants/routes"`, 
     you should read the file at `src/constants/routes.ts` (or similar) completely and pass it as `helpful_context`.
4. **Extract**: Call `extract_routes_from_file` tool with:
   - `file_path`: Full path to router file
   - `file_content`: Complete content of router file
   - `helpful_context`: Content of constant definition files if you found them
5. **Validate & Polish**: After extraction, ALWAYS:
   - Use `get_routes_state` tool to check the current routes stored in memory
   - Check if paths are in correct format (should be strings starting with "/", e.g., "/login")
   - If you see constant patterns like "ROUTES.LOGIN" or "CONSTANTS.HOME" instead of actual paths:
     a. Use `grep_text` to search for constant definitions (e.g., grep for "ROUTES.LOGIN" or "LOGIN:")
     b. Use `find_file` to locate constants files (e.g., "constants", "route.ts")
     c. Read the constant definition files using `read_file`
     d. Call `extract_routes_from_file` again with router file AND the helpful context you found
6. **Final Save**: Once all paths are validated and resolved, use `save_routes_final` tool to write routes.json to disk.
   This tool will perform final validation and only save if all paths are correct.

### OUTPUT FORMAT REQUIREMENTS
The final routes.json must have this exact format:
[
  {{"path": "/login", "component": "src/modules/login/Login.tsx"}},
  {{"path": "/dashboard", "component": "src/modules/home/Home.tsx"}}
]

CRITICAL:
**WHEN `save_routes_final` RETURNS "SUCCESS": YOUR TASK IS COMPLETE. STOP IMMEDIATELY.**
- Path values MUST be actual strings starting with "/" (e.g., "/login", "/home")
- NEVER use constant names like "ROUTES.LOGIN" in the final output
- Component paths should be relative to src directory (e.g., "src/modules/login/Login.tsx")
- Always validate routes.json after extraction and fix any issues before finishing
"""

def _extract_retry_after(error_msg: str) -> int:
    """Extract retry_after seconds from rate limit error message."""
    match = re.search(r'Please try again in (\d+)s', error_msg)
    if match:
        return int(match.group(1))
    return 20  # Default wait time

async def _run_with_retry(graph, inputs, config, logger=None, max_retries=3):
    """
    Run the graph with exponential backoff retry on rate limit errors.
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if logger:
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
                
                return  # Success, exit retry loop
            else:
                # Simple invocation without logging
                result = graph.invoke(inputs, config=config)
                return result
                
        except RateLimitError as e:
            last_error = e
            error_msg = str(e)
            wait_time = _extract_retry_after(error_msg)
            
            if attempt < max_retries - 1:
                print(f"      ⏳ Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s before retry...")
                if logger:
                    logger.add_event("rate_limit_retry", {
                        "attempt": attempt + 1,
                        "wait_time": wait_time,
                        "error": error_msg[:200]
                    })
                await asyncio.sleep(wait_time)
            else:
                print(f"      ❌ Rate limit error after {max_retries} attempts.")
                raise
        except Exception as e:
            # Non-rate-limit errors: re-raise immediately
            raise
    
    # If we exhausted retries, raise the last error
    if last_error:
        raise last_error

@tool("extract_routes_from_file")
def extract_routes_from_file(file_path: str, file_content: str, helpful_context: str = None) -> str:
    """
    Extract route-component pairs from a router configuration file using GPT-4o.
    This tool reads the entire router file and automatically extracts all routes and their components.
    Optionally accepts constants context to resolve constant references to actual paths.
    
    Args:
        file_path: Full path to the router file (e.g., "/path/to/router.tsx")
        file_content: The complete content of the router file as a string.
        helpful_context: (Optional) Content of constant definition files (e.g., ROUTES constants).
                          If provided, the tool will use this to resolve constant references to actual paths.
    
    Returns:
        Success message if routes were extracted and saved, or error message.
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "ERROR: OPENAI_API_KEY environment variable is missing."
        
        client = OpenAI(api_key=api_key)
        
        # Build constants context section if provided
        helpful_context_section = ""
        if helpful_context:
            helpful_context_section = f"""

HELPFUL CONTEXT (use these to resolve constant references to actual paths):
{helpful_context}

IMPORTANT: When you see a path like ROUTES.LOGIN in the router file, look it up in the constants definitions above 
and use the ACTUAL PATH VALUE (e.g., "/login") in the output, NOT the constant name."""

        extraction_prompt = f"""You are a Senior Frontend Software Engineer and React Router expert. Analyze the following router configuration file and extract ALL route-component pairs.

ROUTER FILE PATH: {file_path}

ROUTER FILE CONTENT:
{file_content}{helpful_context_section}

TASK:
1. Identify all route definitions (look for `path:` properties in the router array)
2. For each route, identify the component being rendered (look for JSX elements like `<Home />`, `<Login />`, etc.)
3. Map the route path to the component file path based on the imports at the top of the file
4. **CRITICAL**: If helpful_context is provided above, use it to resolve constant references (e.g., ROUTES.LOGIN -> "/login").
   If helpful_context is NOT provided and you see constants, you may keep them as-is, but prefer actual paths when possible.

### OUTPUT FORMAT
Return a JSON object with a "routes" key containing an array:
{{
  "routes": [
    {{"path": "/login", "component": "src/modules/login/Login.tsx"}},
    {{"path": "/dashboard", "component": "src/modules/home/Home.tsx"}}
  ]
}}

IMPORTANT:
- Extract the component file path from the import statements (e.g., `import Home from "@/modules/home/Home"` means component path is "src/modules/home/Home.tsx")
- **Path values MUST be actual strings starting with "/" (e.g., "/login", "/home") when constants are resolved**
- If constants cannot be resolved, you may keep constant names temporarily, but actual paths are preferred
- Include ALL routes, even nested or protected routes
- Component path should be relative to the src directory (e.g., "src/modules/login/Login.tsx")

Return ONLY valid JSON, no other text."""

        # Call GPT-4o for extraction
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a React Router expert. Extract route-component pairs from router configuration files. Always return valid JSON only."
                },
                {
                    "role": "user",
                    "content": extraction_prompt
                }
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        output_text = response.choices[0].message.content.strip()
        
        # Clean up JSON if wrapped in markdown
        if output_text.startswith("```json"):
            output_text = output_text.replace("```json", "").replace("```", "").strip()
        elif output_text.startswith("```"):
            output_text = output_text.replace("```", "").strip()
        
        # Parse JSON
        try:
            parsed = json.loads(output_text)
            # Extract routes array from the JSON object
            if isinstance(parsed, dict) and "routes" in parsed:
                routes_data = parsed["routes"]
            elif isinstance(parsed, list):
                routes_data = parsed
            else:
                return f"ERROR: Unexpected JSON structure. Expected object with 'routes' key or array, got: {type(parsed).__name__}"
        except json.JSONDecodeError as e:
            # Try to extract JSON array from text
            json_match = re.search(r'\[.*\]', output_text, re.DOTALL)
            if json_match:
                routes_data = json.loads(json_match.group(0))
            else:
                return f"ERROR: Could not parse JSON from model response: {str(e)}\nResponse: {output_text[:500]}"
        
        # Ensure it's a list
        if not isinstance(routes_data, list):
            return f"ERROR: Expected JSON array, got {type(routes_data).__name__}"
        
        # Store in global state instead of writing to file immediately
        global _routes_state
        _routes_state["routes"] = routes_data
        _routes_state["file_path"] = file_path
        _routes_state["extracted_at"] = datetime.datetime.now().isoformat()
        
        # Check if paths need further resolution (contain constant patterns)
        unresolved_constants = []
        for route in routes_data:
            path = route.get("path", "")
            if re.search(r'[A-Z_][A-Z0-9_]*\.[A-Z_][A-Z0-9_]*', path):
                unresolved_constants.append(path)
        
        if unresolved_constants:
            return (f"SUCCESS: Extracted {len(routes_data)} routes from {file_path} and stored in state. "
                   f"However, {len(unresolved_constants)} paths still contain constant references (e.g., {unresolved_constants[0]}). "
                   f"Use 'get_routes_state' tool to check the current routes, then use your tools (grep_text, find_file, read_file) "
                   f"to find constant definitions, and call extract_routes_from_file again with helpful_context parameter to resolve them. "
                   f"Once all paths are resolved, use 'save_routes_final' tool to write routes.json.")
        else:
            return (f"SUCCESS: Extracted {len(routes_data)} routes from {file_path} and stored in state. "
                   f"All paths appear to be resolved. Use 'get_routes_state' tool to verify, then 'save_routes_final' to write routes.json.")
        
    except Exception as e:
        import traceback
        return f"ERROR: Failed to extract routes: {str(e)}\n{traceback.format_exc()[:500]}"

@tool("get_routes_state")
def get_routes_state() -> str:
    """
    Get the current routes stored in memory state. Use this to check routes before final validation.
    Routes are stored in state after extraction and can be updated before final save.
    
    Returns:
        JSON string of the current routes state, or error message if no routes are stored yet.
    """
    global _routes_state
    
    if _routes_state["routes"] is None:
        return "ERROR: No routes in state yet. Call extract_routes_from_file first."
    
    state_info = {
        "routes": _routes_state["routes"],
        "file_path": _routes_state["file_path"],
        "extracted_at": _routes_state["extracted_at"],
        "total_routes": len(_routes_state["routes"])
    }
    
    return json.dumps(state_info, indent=2)

@tool("save_routes_final")
def save_routes_final(output_path: str = "routes.json") -> str:
    """
    Save the routes from state to a JSON file. Only call this after validating that all paths are correct.
    This is the final step - routes.json will be written to the specified path.
    
    Args:
        output_path: Path where to save routes.json (default: "routes.json")
    
    Returns:
        Success message with file path, or error message if no routes in state or validation fails.
    """
    global _routes_state
    
    if _routes_state["routes"] is None:
        return "ERROR: No routes in state. Call extract_routes_from_file first."
    
    routes_data = _routes_state["routes"]
    
    # Final validation: check if any paths still contain constant patterns
    unresolved_constants = []
    invalid_paths = []
    
    for i, route in enumerate(routes_data):
        path = route.get("path", "")
        component = route.get("component", "")
        
        # Check for constant patterns
        if re.search(r'[A-Z_][A-Z0-9_]*\.[A-Z_][A-Z0-9_]*', path):
            unresolved_constants.append(f"Route {i+1}: {path}")
        
        # Check if path is valid format (should start with "/" or be "*")
        if not (path.startswith("/") or path == "*"):
            invalid_paths.append(f"Route {i+1}: {path} (should start with '/' or be '*')")
        
        # Check if component is present
        if not component:
            invalid_paths.append(f"Route {i+1}: missing component")
    
    if unresolved_constants:
        return (f"ERROR: Cannot save routes.json - {len(unresolved_constants)} paths still contain constant references:\n"
               f"{chr(10).join(unresolved_constants[:5])}\n"
               f"Please resolve these constants first using your tools.")
    
    if invalid_paths:
        return (f"ERROR: Cannot save routes.json - {len(invalid_paths)} routes have invalid format:\n"
               f"{chr(10).join(invalid_paths[:5])}\n"
               f"Please fix these issues first.")
    
    # All validation passed - save to file
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(routes_data, f, indent=2)
        
        return f"SUCCESS: Saved {len(routes_data)} validated routes to {os.path.abspath(output_path)}"
        
    except Exception as e:
        return f"ERROR: Failed to save routes.json: {str(e)}"

async def _run_route_discovery_async(enable_logging: bool = False):
    """
    Async helper function that runs the agent with event streaming for logging.
    """
    logger = RouteDiscoveryLogger() if enable_logging else None
    
    tools = [
        LocalTools.find_file,
        LocalTools.grep_text,
        LocalTools.read_file,
        extract_routes_from_file,
        get_routes_state,
        save_routes_final
    ]

    # Use gpt-4o-mini for higher rate limits (500 RPM vs 3 RPM for gpt-4o)
    # Add retry mechanism for rate limit errors
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0,
        max_retries=3,  # LangChain built-in retry
        request_timeout=60
    )
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
    
    config = {"recursion_limit": 20}
    
    try:
        # Use retry wrapper with exponential backoff
        await _run_with_retry(graph, inputs, config, logger=logger, max_retries=3)
            
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
        
    except RateLimitError as e:
        error_msg = str(e)
        print(f"      ❌ Rate Limit Error: {error_msg}")
        print(f"      ⏳ Please wait before retrying. The API rate limit has been reached.")
        
        # Try to save partial results if routes.json exists
        routes_found = None
        if os.path.exists("routes.json"):
            try:
                with open("routes.json", "r", encoding="utf-8") as f:
                    routes_found = json.load(f)
                print(f"      💾 Partial results saved: {len(routes_found)} routes found before rate limit.")
            except Exception as read_err:
                print(f"      ⚠️  Could not read partial results: {read_err}")
        
        if logger:
            logger.add_event("error", {
                "message": error_msg,
                "type": "RateLimitError",
                "partial_routes_count": len(routes_found) if routes_found else 0
            })
            logger.save_log(cwd, False, routes_found)
        
        # Return partial results if available, otherwise return empty list
        return routes_found if routes_found else []
        
    except Exception as e:
        error_msg = str(e)
        print(f"      ❌ Error: {error_msg}")
        
        # Try to save partial results if routes.json exists
        routes_found = None
        if os.path.exists("routes.json"):
            try:
                with open("routes.json", "r", encoding="utf-8") as f:
                    routes_found = json.load(f)
                print(f"      💾 Partial results saved: {len(routes_found)} routes found before error.")
            except Exception as read_err:
                pass
        
        if logger:
            logger.add_event("error", {"message": error_msg, "type": type(e).__name__})
            logger.save_log(cwd, False, routes_found)
        
        # Return partial results if available, otherwise return empty list
        return routes_found if routes_found else []

def run_route_discovery(enable_logging: bool = False):
    """
    Runs a specialized agent to discover routes generically using codebase tools.
    
    Args:
        enable_logging: If True, enables detailed logging to file.
    """
    print("🚀 Spawning Route Discovery Agent...")
    
    if enable_logging:
        # Run async version for detailed logging
        try:
            routes = asyncio.run(_run_route_discovery_async(enable_logging=True))
            if routes:
                print(f"✅ Found {len(routes)} routes.")
            else:
                print("⚠️  No routes found. Check logs for details.")
        except Exception as e:
            print(f"❌ Route discovery failed: {e}")
            print("💾 Check the log directory for partial results.")
            return None
    else:
        # Simple synchronous version
        tools = [
            LocalTools.find_file,
            LocalTools.grep_text,
            LocalTools.read_file,
            extract_routes_from_file,
            get_routes_state,
            save_routes_final
        ]

        # Use gpt-4o-mini for higher rate limits
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0,
            max_retries=3,
            request_timeout=60
        )
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
        
        graph.invoke(inputs, config={"recursion_limit": 20})
    
    print("✅ Route Discovery Process Completed.")