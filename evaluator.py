import os
import subprocess
import asyncio
import time
import json
import datetime
import glob
import ast
from typing import Set, List, Dict, Any
from datasets import load_dataset
from unidiff import PatchSet
import pandas as pd
from langchain_core.messages import HumanMessage

# Imports (Assumed to exist in your environment)
from src.local_tools import LocalTools
from src.agent import build_graph

# --- 1. UPDATED LOGGER: Task-Based Directories & Rotation ---
class BenchmarkLogger:
    def __init__(self, base_dir="./benchmark_logs"):
        self.base_dir = base_dir

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
        files = glob.glob(os.path.join(task_dir, "*.json"))
        files.sort(key=os.path.getmtime)
        
        while len(files) > limit:
            file_to_remove = files.pop(0)
            try:
                os.remove(file_to_remove)
                print(f"      🗑️ Rotated log: {os.path.basename(file_to_remove)}")
            except OSError as e:
                print(f"      ⚠️ Failed to delete old log: {e}")

# --- HELPER: SUBPROCESS ---
def run_command(args, cwd=None, description=""):
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

# --- HELPER: FUNCTION GRANULARITY PARSER ---
def get_gold_functions(patch_text: str, repo_path: str) -> Set[str]:
    """
    Parses the patch and repo to return a set of 'filepath::function_name' strings
    identifying which functions were actually modified by the gold patch.
    """
    gold_funcs = set()
    patch = PatchSet(patch_text)

    for patched_file in patch:
        if patched_file.is_removed_file: continue
        
        # We only support function granularity for Python here
        if not patched_file.path.endswith(".py"):
            gold_funcs.add(patched_file.path)
            continue

        full_path = os.path.join(repo_path, patched_file.path)
        if not os.path.exists(full_path): continue

        try:
            # Parse the file state at base_commit
            with open(full_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            # Identify which lines are changed in the SOURCE (base) file
            changed_lines = set()
            for hunk in patched_file:
                # hunk.source_start is the starting line number in the original file
                for i in range(hunk.source_start, hunk.source_start + hunk.source_length):
                    changed_lines.add(i)

            # Map changed lines to function definitions
            found_in_file = False
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # inclusive range of the function body
                    func_range = range(node.lineno, node.end_lineno + 1)
                    
                    # If the patch touches this function
                    if any(line in func_range for line in changed_lines):
                        gold_funcs.add(f"{patched_file.path}::{node.name}")
                        found_in_file = True
            
            # If changes were outside a function (module level or globals), track the file
            if not found_in_file and changed_lines:
                gold_funcs.add(f"{patched_file.path}::__global__")

        except Exception as e:
            # Fallback if parsing fails
            print(f"      ⚠️ AST Parse Error for {patched_file.path}: {e}")
            gold_funcs.add(patched_file.path)
            
    return gold_funcs

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
        config = {"configurable": {"thread_id": "eval"}, "recursion_limit": 25}
        
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
                    usage = getattr(output, "usage_metadata", {}) or {}

                    if not usage and hasattr(output, "response_metadata"):
                        meta = output.response_metadata
                        usage = meta.get("token_usage") or meta.get("usage") or {}

                    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
                    total = usage.get("total_tokens") or (input_tokens + output_tokens)
                    
                    total_tokens["input"] += input_tokens
                    total_tokens["output"] += output_tokens
                    total_tokens["total"] += total
                    
                    if model_signature == "unknown_model":
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
            
            # --- UPDATED: Calculate Function-Level Targets ---
            gold_targets = get_gold_functions(row['patch'], instance_path)
            print(f"      🎯 Targets: {len(gold_targets)} functions identified")

            start_time = time.time()
            agent_files_raw = await run_retrieval_session(
                instance_path, 
                row['problem_statement'], 
                trace_bucket=trace_history
            )
            duration = time.time() - start_time
            
            # --- PROCESS STATS ---
            run_stats = {"tokens": {}, "model": "unknown"}
            if trace_history and trace_history[-1].get("type") == "__META_STATS__":
                meta = trace_history.pop() 
                run_stats = meta
            
            # --- UPDATED: Evaluate Matches (Function Level) ---
            agent_files = {f.lstrip("./") for f in agent_files_raw}
            
            matches = []
            for target in gold_targets:
                # Target format: "path/to/file.py::func_name"
                # Check if the file containing the function was read by the agent
                target_file = target.split("::")[0]
                if target_file in agent_files:
                    matches.append(target)
            
            success = len(matches) > 0
            
            print(f"   {'✅' if success else '❌'} Result: {len(matches)}/{len(gold_targets)} | Tokens: {run_stats['tokens'].get('total', 0)}")

            # --- CONSTRUCT LOG ---
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
                    "gold_granularity": "function",
                    "gold_targets": list(gold_targets),
                    "base_commit": row['base_commit']
                },
                "agent_output": {
                    "files_found": list(agent_files),
                    "matches_found": matches,
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
