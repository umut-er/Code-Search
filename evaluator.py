import os
import subprocess
import asyncio
import time
import json
import datetime
import glob
from typing import Set, List, Dict, Any
from datasets import load_dataset
from unidiff import PatchSet
import pandas as pd
from langchain_core.messages import HumanMessage

# Imports
from src.local_tools import LocalTools
from src.agent import build_graph

# --- 1. UPDATED LOGGER: Task-Based Directories & Rotation ---
class BenchmarkLogger:
    def __init__(self, base_dir="./benchmark_logs"):
        self.base_dir = base_dir
        # We don't create a run folder here anymore. 
        # Folders are created per-task.

    def save_task_log(self, instance_id: str, data: Dict):
        """
        Saves log to ./benchmark_logs/{instance_id}/{timestamp}.json
        and enforces the 'keep last 3' rule.
        """
        # 1. Create Task Directory
        task_dir = os.path.join(self.base_dir, instance_id)
        os.makedirs(task_dir, exist_ok=True)

        # 2. Generate Filename with Timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join(task_dir, f"{timestamp}.json")

        # 3. Save JSON
        def set_default(obj):
            if isinstance(obj, set):
                return list(obj)
            return str(obj)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=set_default)
        
        # 4. Enforce Retention Policy (Keep Last 3)
        self._rotate_logs(task_dir, limit=3)

    def _rotate_logs(self, task_dir: str, limit: int):
        """Deletes oldest files if count > limit."""
        # Get all json files in the directory
        files = glob.glob(os.path.join(task_dir, "*.json"))
        
        # Sort by modification time (oldest first)
        files.sort(key=os.path.getmtime)
        
        # Delete if we have too many
        while len(files) > limit:
            file_to_remove = files.pop(0)
            try:
                os.remove(file_to_remove)
                print(f"      🗑️ Rotated log: {os.path.basename(file_to_remove)}")
            except OSError as e:
                print(f"      ⚠️ Failed to delete old log: {e}")

# --- HELPER: SUBPROCESS ---
def run_command(args, cwd=None, description=""):
    start = time.time()
    try:
        subprocess.run(
            args, 
            cwd=cwd, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True
        )
    except subprocess.CalledProcessError:
        print(f" ❌ Failed to run: {' '.join(args)}")
        raise

def get_gold_files(patch_text: str) -> Set[str]:
    patch = PatchSet(patch_text)
    return {p.path for p in patch}

# --- 2. MODIFIED SESSION: Captures Tokens & Model Info ---
async def run_retrieval_session(repo_path: str, issue_text: str, trace_bucket: list = None) -> Set[str]:
    original_cwd = os.getcwd()
    os.chdir(repo_path) 
    
    read_files = set()
    
    # Track usage locally
    total_tokens = {"input": 0, "output": 0, "total": 0}
    model_signature = "unknown_model"

    try:
        tools = [
            LocalTools.read_file, LocalTools.list_directory,
            LocalTools.search_code, LocalTools.get_code_symbols
        ]
        
        graph = build_graph(tools)
        config = {"configurable": {"thread_id": "eval"}, "recursion_limit": 15}
        
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=f"Locate the code responsible for: {issue_text}")]},
            config=config,
            version="v2"
        ):
            kind = event["event"]
            data = event["data"]
            
            # --- METADATA EXTRACTION ---
            if kind == "on_chat_model_end":
                output = data.get("output")
                if output:
                    # --- 1. UPDATED: LangChain v1 Standardized Usage ---
                    # usage_metadata is the new standard (v0.3+ / v1.0)
                    # It normalizes keys to: input_tokens, output_tokens, total_tokens
                    usage = getattr(output, "usage_metadata", {}) or {}

                    # Fallback for older models/providers not yet adhering to v1 standard
                    if not usage and hasattr(output, "response_metadata"):
                        meta = output.response_metadata
                        usage = meta.get("token_usage") or meta.get("usage") or {}

                    # --- 2. Extract Counts (Robustly) ---
                    # v1 uses "input_tokens", OpenAI legacy uses "prompt_tokens"
                    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
                    total = usage.get("total_tokens") or (input_tokens + output_tokens)
                    
                    total_tokens["input"] += input_tokens
                    total_tokens["output"] += output_tokens
                    total_tokens["total"] += total
                    
                    # --- 3. Capture Model Name ---
                    if model_signature == "unknown_model":
                         # Try v1 standard .name or fallback to metadata
                         model_signature = getattr(output, "name", None) or \
                                           output.response_metadata.get("model_name") or \
                                           output.response_metadata.get("model") or \
                                           "unknown_model"

            # --- LOGGING INTERCEPTOR ---
            if trace_bucket is not None:
                if kind == "on_chat_model_end":
                    content = data.get("output", {}).content
                    if content:
                        trace_bucket.append({
                            "type": "thought", 
                            "content": content, 
                            "timestamp": time.time()
                        })
                elif kind == "on_tool_start":
                    trace_bucket.append({
                        "type": "tool_call",
                        "tool": event["name"],
                        "input": data.get("input"),
                        "timestamp": time.time()
                    })
                elif kind == "on_tool_end":
                    trace_bucket.append({
                        "type": "tool_result",
                        "tool": event["name"],
                        "output": data.get("output"),
                        "timestamp": time.time()
                    })
            
            # ORIGINAL CONSOLE LOGGING
            if kind == "on_tool_start":
                tool_name = event["name"]
                if tool_name == "read_file":
                    path = event["data"].get("input", {}).get("path", "")
                    print(f"      👀 Reading: {path}")
                    read_files.add(path)
                elif tool_name == "search_code_lexical":
                    query = event["data"].get("input", {}).get("query", "")
                    print(f"      🔎 Searching: '{query}'")

    except Exception as e:
        if trace_bucket is not None:
            trace_bucket.append({"type": "error", "message": str(e)})
        
        if "recursion limit" in str(e).lower():
            print(f"      ⚠️  Hit Step Limit (15).")
        else:
            print(f"      ❌ Agent Error: {e}")
    finally:
        os.chdir(original_cwd)
        # Inject the collected stats into the trace bucket as a hidden "meta" item
        # We will extract this in the main loop before saving.
        if trace_bucket is not None:
            trace_bucket.append({
                "type": "__META_STATS__", 
                "tokens": total_tokens, 
                "model": model_signature
            })

    return read_files

async def evaluate_swe_lite(limit=5):
    print("📥 Loading SWE-bench Lite dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    
    logger = BenchmarkLogger()
    work_dir = os.path.abspath("./benchmark_scratch")
    os.makedirs(work_dir, exist_ok=True)
    
    count = 0
    results = []

    for row in dataset:
        if count >= limit: break
        
        instance_id = row['instance_id']
        repo_url = f"https://github.com/{row['repo']}"
        repo_name = row['repo'].split("/")[-1]
        base_commit = row['base_commit']
        
        print(f"\n[{count+1}/{limit}] 🧪 {instance_id}")
        
        instance_path = os.path.join(work_dir, repo_name)
        trace_history = [] 

        try:
            if not os.path.exists(instance_path):
                print(f"   ⬇️  Cloning {repo_name}...")
                run_command(["git", "clone", repo_url, instance_path])
            
            # Reset
            run_command(["git", "reset", "--hard", "HEAD"], cwd=instance_path)
            run_command(["git", "clean", "-fdx"], cwd=instance_path)
            run_command(["git", "checkout", base_commit], cwd=instance_path)
            
            gold_files = get_gold_files(row['patch'])

            start_time = time.time()
            agent_files_raw = await run_retrieval_session(
                instance_path, 
                row['problem_statement'], 
                trace_bucket=trace_history
            )
            duration = time.time() - start_time
            
            # --- PROCESS STATS ---
            # Extract the hidden metadata we pushed at the end of the session
            run_stats = {"tokens": {}, "model": "unknown"}
            if trace_history and trace_history[-1].get("type") == "__META_STATS__":
                meta = trace_history.pop() # Remove it from the history list
                run_stats = meta
            
            agent_files = {f.lstrip("./") for f in agent_files_raw}
            matches = gold_files.intersection(agent_files)
            success = len(matches) > 0
            
            print(f"   {'✅' if success else '❌'} Result: {len(matches)}/{len(gold_files)} | Tokens: {run_stats['tokens'].get('total', 0)}")

            # --- CONSTRUCT LOG (Ordered) ---
            # We explicitly construct the dictionary to keep LLM signature at top
            log_data = {
                "llm_signature": {
                    "model": run_stats["model"],
                    "token_usage": run_stats["tokens"],
                    "timestamp": datetime.datetime.now().isoformat()
                },
                "meta": {
                    "instance_id": instance_id,
                    "success": success,
                    "duration_seconds": duration,
                },
                "task": {
                    "problem_statement": row['problem_statement'],
                    "gold_files": list(gold_files),
                    "base_commit": row['base_commit']
                },
                "agent_output": {
                    "files_found": list(agent_files),
                    "history": trace_history
                }
            }
            
            logger.save_task_log(instance_id, log_data)

            results.append({
                "id": instance_id, "success": success, "matches": len(matches)
            })
            count += 1
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            logger.save_task_log(instance_id, {
                "llm_signature": "CRASH", 
                "error": str(e), 
                "history": trace_history
            })
            continue

    if results:
        df = pd.DataFrame(results)
        print("\n\n📊 ================= SUMMARY =================")
        print(df[["id", "success", "matches"]])
        print(f"🏆 Accuracy: {df['success'].mean() * 100:.2f}%")
        print("==========================================")

if __name__ == "__main__":
    asyncio.run(evaluate_swe_lite(limit=5))
