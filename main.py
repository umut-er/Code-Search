import os
import asyncio
import time
import json
import datetime
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Import from src
from src.local_tools import LocalTools
from src.agent import build_graph
from src.s2r_tool import generate_s2r

load_dotenv()

# --- LOGGER SETUP ---
class SessionLogger:
    def __init__(self, base_dir="./interactive_logs"):
        self.base_dir = base_dir
        self.session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(os.path.join(self.base_dir, self.session_id), exist_ok=True)
        print(f"📁 Logging session to: {os.path.join(self.base_dir, self.session_id)}")

    def save_turn(self, turn_index: int, user_input: str, trace: list, stats: dict):
        """Saves a JSON log for a single interaction turn."""
        filename = f"turn_{turn_index:03d}.json"
        filepath = os.path.join(self.base_dir, self.session_id, filename)
        
        log_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_input": user_input,
            "llm_signature": stats.get("model", "unknown"),
            "token_usage": stats.get("tokens", {}),
            "trace": trace
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, default=str)

async def run_interactive():
    print("🚀 Starting Agent (Interactive Mode)...")
    logger = SessionLogger()
    
    # 1. Load Tools (The Full Suite)
    tools = [
        # --- Code Navigation / Context Retrieval ---
        LocalTools.list_directory,      # Navigation
        LocalTools.read_file,           # Reading Content
        LocalTools.find_file,           # Fuzzy Filename Search 
        LocalTools.find_usage,          # Upward Traversal 
        LocalTools.read_file_skeleton,  # High-level structure
        LocalTools.grep_text,           # Lexical Search

        # --- Bug → S2R Pipeline ---
        # This tool takes a raw bug report + a set of relevant code paths
        # and returns a JSON S2R script suitable for a browser agent.
        generate_s2r,
    ]
    
    # 2. Build Graph
    graph = build_graph(tools)
    
    # Thread ID persists context across the session
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}

    print(f"🤖 Ready. Session ID: {logger.session_id}")
    print("Type 'exit' to quit.\n")
    
    turn_count = 0
    
    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input:
                continue
            
            turn_count += 1
            trace_bucket = []
            
            # Reset stats for this turn
            turn_stats = {
                "tokens": {"input": 0, "output": 0, "total": 0}, 
                "model": "unknown_model"
            }
            
            print("\n--- Agent Execution ---")

            async for event in graph.astream_events(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                version="v2"
            ):
                kind = event["event"]
                data = event["data"]
                name = event["name"]

                # --- 1. CAPTURE METADATA (Tokens & Model) ---
                if kind == "on_chat_model_end":
                    output = data.get("output")
                    if output:
                        # Extract Usage
                        usage = getattr(output, "usage_metadata", {}) or {}
                        if not usage and hasattr(output, "response_metadata"):
                            meta = output.response_metadata
                            usage = meta.get("token_usage") or meta.get("usage") or {}

                        i_tok = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                        o_tok = usage.get("output_tokens") or usage.get("completion_tokens") or 0
                        total = usage.get("total_tokens") or (i_tok + o_tok)
                        
                        turn_stats["tokens"]["input"] += i_tok
                        turn_stats["tokens"]["output"] += o_tok
                        turn_stats["tokens"]["total"] += total

                        # Extract Model Name
                        if turn_stats["model"] == "unknown_model":
                             turn_stats["model"] = getattr(output, "name", None) or \
                                                   output.response_metadata.get("model_name") or \
                                                   output.response_metadata.get("model") or \
                                                   "unknown_model"

                # --- 2. CAPTURE TRACE (History) ---
                if kind == "on_chat_model_stream":
                    # Stream thought to console
                    chunk = data.get("chunk")
                    if hasattr(chunk, "content") and chunk.content:
                        print(chunk.content, end="", flush=True)

                elif kind == "on_chat_model_end":
                    content = data.get("output", {}).content
                    if content:
                        print() # Newline after streaming
                        trace_bucket.append({
                            "type": "thought", 
                            "content": content, 
                            "timestamp": time.time()
                        })

                elif kind == "on_tool_start":
                    tool_input = data.get("input") or {}
                    print(f"\n🛠️  Calling {name}...")
                    
                    # Console hints for specific tools
                    if name == "find_file":
                        print(f"    Searching name_pattern={tool_input.get('name_pattern')!r} in path={tool_input.get('path')!r}")
                    elif name == "grep_text":
                        print(f"    Grepping query={tool_input.get('query')!r} in path={tool_input.get('path')!r}")
                    elif name == "list_directory":
                        print(f"    Listing directory path={tool_input.get('path')!r}")
                    elif name == "read_file":
                        print(
                            f"    Reading file path={tool_input.get('path')!r} "
                            f"lines={tool_input.get('start_line')}..{tool_input.get('end_line')}"
                        )
                    elif name == "read_file_skeleton":
                        print(f"    Skeleton view for path={tool_input.get('path')!r}")
                    elif name == "find_usage":
                        print(
                            f"    Finding usage of filename={tool_input.get('filename')!r} "
                            f"in path={tool_input.get('path')!r}"
                        )
                    elif name == "generate_s2r":
                        bug_preview = str(tool_input.get('bug_report', ''))[:80].replace("\n", " ")
                        print(
                            "    Generating S2R with code_paths="
                            f"{tool_input.get('code_paths')!r} "
                            f"for bug_report≈{bug_preview!r}"
                        )
                    
                    trace_bucket.append({
                        "type": "tool_call",
                        "tool": name,
                        "input": tool_input,
                        "timestamp": time.time()
                    })

                elif kind == "on_tool_end":
                    output_data = data.get("output")
                    output_str = str(output_data)
                    
                    # Truncate for log if massive
                    log_output = output_str
                    if len(log_output) > 2000:
                        log_output = log_output[:2000] + "... [TRUNCATED]"
                    
                    trace_bucket.append({
                        "type": "tool_result",
                        "tool": name,
                        "output": log_output,
                        "timestamp": time.time()
                    })

            # --- 3. END OF TURN SUMMARY ---
            print(
                f"\n\n[Turn Stats] In: {turn_stats['tokens']['input']} | "
                f"Out: {turn_stats['tokens']['output']} | "
                f"Total: {turn_stats['tokens']['total']}"
            )
            
            # Save Log
            logger.save_turn(turn_count, user_input, trace_bucket, turn_stats)
            print(f"💾 Log saved for Turn {turn_count}.\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            # Ensure we don't lose logs when an error (e.g., rate limit) occurs.
            print(f"\n❌ Critical Error during Turn {turn_count}: {e}")
            import traceback
            traceback.print_exc()
            try:
                # Attach an explicit error entry to the trace and persist it.
                error_entry = {
                    "type": "error",
                    "error": str(e),
                    "timestamp": time.time(),
                }
                # trace_bucket / turn_stats may not exist if failure is very early,
                # so guard with defaults.
                if "trace_bucket" in locals():
                    trace_bucket.append(error_entry)
                else:
                    trace_bucket = [error_entry]
                if "turn_stats" not in locals():
                    turn_stats = {
                        "tokens": {"input": 0, "output": 0, "total": 0},
                        "model": "unknown_model",
                    }
                if "user_input" not in locals():
                    user_input = ""
                logger.save_turn(turn_count, user_input, trace_bucket, turn_stats)
                print(f"💾 Log saved for Turn {turn_count} (with error).\n")
            except Exception:
                # Last-resort: avoid crashing logger path as well.
                pass
            # Continue interactive loop so the user can try again later.
            continue

if __name__ == "__main__":
    asyncio.run(run_interactive())
