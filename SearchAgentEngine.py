import os
import time
import json
import datetime
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Import from src (Mevcut yapınla uyumlu)
from src.local_tools import LocalTools
from src.agent import build_graph

load_dotenv()

# --- LOGGER SETUP (Incremental Update) ---
class SessionLogger:
    def __init__(self, base_dir="./interactive_logs"):
        self.base_dir = base_dir
        self.session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(os.path.join(self.base_dir, self.session_id), exist_ok=True)
        print(f"📁 Logging session to: {os.path.join(self.base_dir, self.session_id)}")
        self.turn_counter = 0
        self.current_turn_filepath = None
        self.current_turn_data = None

    def start_turn(self, user_input: str):
        """Initialize a new turn and create the JSON file."""
        self.turn_counter += 1
        filename = f"turn_{self.turn_counter:03d}.json"
        self.current_turn_filepath = os.path.join(self.base_dir, self.session_id, filename)
        
        self.current_turn_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "turn_index": self.turn_counter,
            "user_input": user_input,
            "llm_signature": "unknown",
            "token_usage": {"input": 0, "output": 0, "total": 0},
            "trace": []
        }
        
        # Create initial file
        self._write_turn_file()

    def update_turn_trace(self, trace_entry: dict):
        """Add a new entry to trace and immediately update the JSON file."""
        if self.current_turn_data is None:
            return  # Turn not started yet
        
        self.current_turn_data["trace"].append(trace_entry)
        self._write_turn_file()

    def update_turn_stats(self, stats: dict):
        """Update stats (model, tokens) and immediately update the JSON file."""
        if self.current_turn_data is None:
            return
        
        if stats.get("model") and stats["model"] != "unknown_model":
            self.current_turn_data["llm_signature"] = stats.get("model", "unknown")
        
        tokens = stats.get("tokens", {})
        self.current_turn_data["token_usage"] = {
            "input": tokens.get("input", 0),
            "output": tokens.get("output", 0),
            "total": tokens.get("total", 0)
        }
        self._write_turn_file()

    def _write_turn_file(self):
        """Write current turn data to JSON file (atomic write)."""
        if self.current_turn_filepath and self.current_turn_data:
            # Use atomic write: write to temp file, then rename
            temp_filepath = self.current_turn_filepath + ".tmp"
            try:
                with open(temp_filepath, "w", encoding="utf-8") as f:
                    json.dump(self.current_turn_data, f, indent=2, default=str)
                os.replace(temp_filepath, self.current_turn_filepath)
            except Exception as e:
                print(f"⚠️  Warning: Failed to write log file: {e}")

    def finalize_turn(self, stats: dict):
        """Final update with complete stats before turn ends."""
        self.update_turn_stats(stats)

# --- AGENT ENGINE CLASS ---
class SearchAgentEngine:
    def __init__(self):
        print("🚀 Initializing Agent Engine...")
        self.logger = SessionLogger()
        
        # 1. Load Tools
        tools = [
            LocalTools.list_directory,
            LocalTools.read_file,
            LocalTools.find_file,
            LocalTools.find_usage,
            LocalTools.read_file_skeleton,
            LocalTools.grep_text,
            LocalTools.find_best_route,
            LocalTools.submit_final_report,
        ]
        
        # 2. Build Graph (Sadece bir kez initialize edilir, performans artar)
        self.graph = build_graph(tools)
        print(f"🤖 Agent Ready. Session ID: {self.logger.session_id}")

    async def process_request(self, message: str, thread_id: str = None):
        """
        Pipeline'dan çağıracağın ana fonksiyon budur.
        Input beklemez, parametre olarak alır.
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4()) # Eğer thread verilmezse yeni oluştur

        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
        
        turn_stats = {
            "tokens": {"input": 0, "output": 0, "total": 0}, 
            "model": "unknown_model"
        }
        
        final_response = ""

        print(f"\n--- Agent Processing: {message[:50]}... ---")

        # Start turn logging
        self.logger.start_turn(message)

        try:
            async for event in self.graph.astream_events(
                {"messages": [HumanMessage(content=message)]},
                config=config,
                version="v2"
            ):
                kind = event["event"]
                data = event["data"]
                name = event["name"]

                # --- 1. METADATA CAPTURE ---
                if kind == "on_chat_model_end":
                    output = data.get("output")
                    if output:
                        usage = getattr(output, "usage_metadata", {}) or {}
                        if not usage and hasattr(output, "response_metadata"):
                            meta = output.response_metadata
                            usage = meta.get("token_usage") or meta.get("usage") or {}

                        i_tok = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                        o_tok = usage.get("output_tokens") or usage.get("completion_tokens") or 0
                        turn_stats["tokens"]["input"] += i_tok
                        turn_stats["tokens"]["output"] += o_tok
                        turn_stats["tokens"]["total"] += (i_tok + o_tok)

                        if turn_stats["model"] == "unknown_model":
                             turn_stats["model"] = getattr(output, "name", None) or \
                                                   output.response_metadata.get("model_name") or \
                                                   output.response_metadata.get("model") or \
                                                   "unknown_model"
                        
                        # Update stats in log immediately
                        self.logger.update_turn_stats(turn_stats)

                # --- 2. TRACE CAPTURE ---
                if kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if hasattr(chunk, "content") and chunk.content:
                        # Konsola canlı basmak istersen burayı açabilirsin, 
                        # pipeline içinde genelde sessiz olması tercih edilir.
                        print(chunk.content, end="", flush=True)
                        final_response += chunk.content

                elif kind == "on_chat_model_end":
                    content = data.get("output", {}).content
                    if content:
                        print() 
                        trace_entry = {
                            "type": "thought", 
                            "content": content, 
                            "timestamp": time.time()
                        }
                        self.logger.update_turn_trace(trace_entry)

                elif kind == "on_tool_start":
                    tool_input = data.get("input") or {}
                    print(f"🛠️  Calling {name}...") 
                    trace_entry = {
                        "type": "tool_call",
                        "tool": name,
                        "input": tool_input,
                        "timestamp": time.time()
                    }
                    self.logger.update_turn_trace(trace_entry)

                elif kind == "on_tool_end":
                    output_data = data.get("output")
                    log_output = str(output_data)
                    if len(log_output) > 2000:
                        log_output = log_output[:2000] + "... [TRUNCATED]"
                    
                    trace_entry = {
                        "type": "tool_result",
                        "tool": name,
                        "output": log_output,
                        "timestamp": time.time()
                    }
                    # Update log immediately after tool completes
                    self.logger.update_turn_trace(trace_entry)

            # --- 3. FINALIZE LOGS & RETURN ---
            self.logger.finalize_turn(turn_stats)
            print(f"\n💾 Log finalized. Turn Stats: {turn_stats['tokens']['total']} tokens.")
            
            return {
                "status": "success",
                "response": final_response,
                "stats": turn_stats,
                "thread_id": thread_id
            }

        except Exception as e:
            print(f"\n❌ Error: {e}")
            error_entry = {"type": "error", "error": str(e), "timestamp": time.time()}
            self.logger.update_turn_trace(error_entry)
            self.logger.finalize_turn(turn_stats)
            return {"status": "error", "error": str(e)}